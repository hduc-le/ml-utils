import logging
import time
from typing import List, Union

import pandas as pd
from google.api_core.exceptions import GoogleAPIError
from google.cloud import bigquery, bigquery_storage

from .log_helper import logger_func

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(filename)s - %(funcName)s - %(levelname)s - %(message)s",
)


def bigquery_client(project_id: str) -> bigquery.Client:
    """
    Create a BigQuery client.

    Args:
        project_id (str): The Google Cloud project ID.

    Returns:
        bigquery.Client: The BigQuery client.
    """
    return bigquery.Client(project=project_id)


class BigqueryHelper:
    """
    BigQuery helper class to streamline common tasks such as executing queries,
    retrieving table schemas, listing tables, and more, with additional flexibility and error handling.
    """

    def __init__(
        self,
        project_id: str = None,
        dataset_name: str = None,
        max_wait_seconds: int = 900,
        credentials: str = None,
    ):
        self.project_id = project_id
        self.dataset_name = dataset_name
        self.max_wait_seconds = max_wait_seconds
        self.credentials = credentials
        self.client = bigquery.Client(
            project=project_id, credentials=credentials
        )
        self.dataset = None
        self.table_cache = {}  # {table name (str): table object}
        self.total_gb_used_net_cache = 0
        self.BYTES_PER_GB = 2**30

    def set_project(self, project_id):
        """Set a new project."""
        self.project_id = project_id
        self.client = bigquery.Client(
            project=self.project_id, credentials=self.credentials
        )

    def set_dataset(self, dataset_name):
        """Set a new dataset."""
        self.dataset_name = dataset_name
        self.dataset = None  # Reset dataset to lazy load on next call
        self.table_cache.clear()

    def parse_table_identifier(self, identifier):
        """
        Parse a table identifier of the format 'project_id.dataset_name.table_name'.
        Sets project_id and dataset_name accordingly.
        """
        parts = identifier.split(".")
        if len(parts) == 3:
            project, dataset, table = parts
            if project != self.project_id:
                self.set_project(project)
            if dataset != self.dataset_name:
                self.set_dataset(dataset)
            return table
        elif len(parts) == 2 and self.project_id:
            dataset, table = parts
            if dataset != self.dataset_name:
                self.set_dataset(dataset)
            return table
        else:
            raise ValueError(
                "Table identifier must be in the format 'project_id.dataset_name.table_name'."
            )

    def __fetch_dataset(self):
        """Lazy loading of dataset information."""
        if self.dataset is None:
            self.dataset = self.client.get_dataset(self.dataset_name)

    def __fetch_table(self, identifier: str):
        """
        Fetch a table using a full identifier (project.dataset.table).
        Parses and sets project and dataset if necessary.
        """
        table_name = self.parse_table_identifier(identifier)
        self.__fetch_dataset()
        if table_name not in self.table_cache:
            table_ref = self.client.dataset(self.dataset_name).table(
                table_name
            )
            self.table_cache[table_name] = self.client.get_table(table_ref)

    def list_tables(self):
        """List all tables in the dataset."""
        self.__fetch_dataset()
        return [
            table.table_id for table in self.client.list_tables(self.dataset)
        ]

    def table_schema(self, identifier: str) -> pd.DataFrame:
        """Get a table's schema with nested fields unrolled.

        Args:
            identifier (str): The table identifier in 'project.dataset.table' format.
        Returns:
            pd.DataFrame: The table schema with nested fields unrolled.
        """
        self.__fetch_table(identifier)
        table_name = self.parse_table_identifier(identifier)
        schema = pd.DataFrame.from_dict(
            [
                field.to_api_repr()
                for field in self.table_cache[table_name].schema
            ]
        )
        return self.__unroll_nested_fields(schema)

    def __unroll_nested_fields(self, schema) -> pd.DataFrame:
        """Flatten nested schema fields for easier querying."""
        schema_details = []
        schema.apply(
            lambda field: self.__handle_field(field, schema_details), axis=1
        )
        result = pd.concat(
            [pd.DataFrame.from_dict(field) for field in schema_details]
        )
        return result[["name", "type", "mode"]]

    def __handle_field(self, field, schema_details, top_level_name=""):
        """Helper to unpack nested fields."""
        name = (
            f"{top_level_name}.{field['name']}"
            if top_level_name
            else field["name"]
        )
        schema_details.append(
            [{"name": name, "type": field["type"], "mode": field["mode"]}]
        )
        if isinstance(field.get("fields"), list):
            for subfield in field["fields"]:
                self.__handle_field(subfield, schema_details, name)

    def estimate_query_size(self, query: str):
        """Estimate the data scanned by the query in gigabytes."""
        job_config = bigquery.QueryJobConfig(dry_run=True)
        try:
            query_job = self.client.query(query, job_config=job_config)
            return query_job.total_bytes_processed / self.BYTES_PER_GB
        except GoogleAPIError as e:
            print(f"Failed to estimate query size: {e}")
            return None

    @logger_func(call_depth=0)
    def query_to_dataframe(
        self, query: str, timeout: int = 7776000, verbose: bool = False
    ) -> Union[pd.DataFrame, None]:
        """Execute a query and return the result as a DataFrame.

        Args:
            query (str): The SQL query to execute.
            timeout (int): The maximum time to wait for the query to complete.
            verbose (bool): Whether to show progress bar.
        Returns:
            Union[pd.DataFrame, None]: The query result as a DataFrame, or None if the query fails.
        """
        job_config = bigquery.QueryJobConfig()
        query_job = self.client.query(query, job_config=job_config)
        start_time = time.time()
        try:
            while not query_job.done():
                if (time.time() - start_time) > (
                    timeout or self.max_wait_seconds
                ):
                    print("Max wait time exceeded, cancelling query.")
                    query_job.cancel()
                    return None
                time.sleep(0.1)
            if query_job.total_bytes_billed:
                self.total_gb_used_net_cache += (
                    query_job.total_bytes_billed / self.BYTES_PER_GB
                )
            return query_job.to_dataframe(
                progress_bar_type="tqdm" if verbose else None,
            )
        except GoogleAPIError as e:
            print(f"Query failed: {e}")
            return None

    @logger_func(call_depth=0)
    def query_to_dataframe_safe(
        self, query: str, max_gb_scanned: int = 10
    ) -> Union[pd.DataFrame, None]:
        """Execute a query only if it scans less than max_gb_scanned of data.

        Args:
            query (str): The SQL query to execute.
            max_gb_scanned (int): The maximum data size the query is allowed to scan.
        Returns:
            Union[pd.DataFrame, None]: The query result as a DataFrame, or None if the query fails.
        """
        estimated_size = self.estimate_query_size(query)
        if estimated_size is None or estimated_size > max_gb_scanned:
            print(
                f"Query exceeds max scan size of {max_gb_scanned} GB (estimated: {estimated_size} GB)"
            )
            return None
        return self.query_to_dataframe(query)

    @logger_func(call_depth=0)
    def query_to_dataframe_iterable(self, query: str) -> pd.DataFrame:
        """Execute a query and return the result as a DataFrame iterable."""
        return (
            self.client.query(query)
            .result()
            .to_dataframe_iterable(
                bqstorage_client=bigquery_storage.BigQueryReadClient()
            )
        )

    @logger_func(call_depth=0)
    def query_to_table(self, query: str, table_id: str) -> str:
        """Run a SQL query and save the result as a table in BigQuery.

        Args:
            query (str): The SQL query to execute.
            table_id (str): The table identifier in 'project.dataset.table' format.
        Returns:
            str: The table identifier of the saved table.
        """
        job_config = bigquery.QueryJobConfig(
            destination=table_id, write_disposition="WRITE_TRUNCATE"
        )
        query_job = self.client.query(query, job_config=job_config)
        query_job.result()
        return table_id

    @logger_func(call_depth=0)
    def query_to_csv(self, query: str, gcs_path: str) -> str:
        """Run a SQL query and save the result as a CSV file in Google Cloud Storage.

        Args:
            query (str): The SQL query to execute.
            gcs_path (str): The GCS path to save the CSV file.
        Returns:
            str: The GCS path of the saved CSV file.
        """
        job_config = bigquery.QueryJobConfig(
            destination=gcs_path, write_disposition="WRITE_TRUNCATE"
        )
        query_job = self.client.query(query, job_config=job_config)
        query_job.result()
        return gcs_path

    @logger_func(call_depth=0)
    def query_table_to_dataframe(
        self, table_id: str, verbose: bool = False
    ) -> pd.DataFrame:
        """Read a BigQuery table and return the result as a pandas DataFrame.

        Args:
            table_id (str): The table identifier in 'project.dataset.table' format.
            verbose (bool): Whether to show progress bar.
        Returns:
            pd.DataFrame: The table data as a DataFrame.
        """
        return self.client.query(f"SELECT * FROM {table_id}").to_dataframe(
            progress_bar_type="tqdm" if verbose else None,
            bqstorage_client=bigquery_storage.BigQueryReadClient(),
        )

    def head(
        self,
        identifier: str,
        num_rows: int = 5,
        start_index: int = 0,
        selected_columns: List[str] = None,
    ) -> pd.DataFrame:
        """Fetch the first few rows of a table as a DataFrame.

        Args:
            identifier (str): The table identifier in 'project.dataset.table' format.
            num_rows (int): The number of rows to fetch.
            start_index (int): The starting row index.
            selected_columns (List[str]): The columns to select.
        Returns:
            pd.DataFrame: The first few rows of the table.
        """
        table_name = self.parse_table_identifier(identifier)
        self.__fetch_table(identifier)
        table = self.table_cache[table_name]
        selected_fields = None
        if selected_columns:
            selected_fields = [
                field
                for field in table.schema
                if field.name in selected_columns
            ]
        rows = self.client.list_rows(
            table,
            selected_fields=selected_fields,
            max_results=num_rows,
            start_index=start_index,
        )
        results = [x for x in rows]
        return pd.DataFrame(
            data=[list(x.values()) for x in results],
            columns=list(results[0].keys()),
        )

    def delete_table(self, identifier: str):
        """Delete a table using a full identifier (project.dataset.table).

        Args:
            identifier (str): The table identifier in 'project.dataset.table' format
        """
        table_name = self.parse_table_identifier(identifier)
        table_ref = self.client.dataset(self.dataset_name).table(table_name)
        try:
            self.client.delete_table(table_ref)
            print(f"Table {table_name} deleted.")
        except GoogleAPIError as e:
            print(f"Failed to delete table {table_name}: {e}")

    def write_to_table(self, df, identifier, if_exists="fail"):
        """
        Write a DataFrame to a BigQuery table.

        Args:
            df (pd.DataFrame): The DataFrame to write.
            identifier (str): The table identifier in 'project.dataset.table' format.
            if_exists (str): Behavior when table exists: 'fail', 'replace', or 'append'.
        """
        table_name = self.parse_table_identifier(identifier)
        table_ref = self.client.dataset(self.dataset_name).table(table_name)

        write_disposition = {
            "fail": bigquery.WriteDisposition.WRITE_EMPTY,
            "replace": bigquery.WriteDisposition.WRITE_TRUNCATE,
            "append": bigquery.WriteDisposition.WRITE_APPEND,
        }.get(if_exists, bigquery.WriteDisposition.WRITE_EMPTY)

        job_config = bigquery.LoadJobConfig(
            write_disposition=write_disposition
        )

        try:
            load_job = self.client.load_table_from_dataframe(
                df, table_ref, job_config=job_config
            )
            load_job.result()  # Wait for job to complete
            print(
                f"Data written to table {table_name} with disposition '{if_exists}'."
            )
        except GoogleAPIError as e:
            print(f"Failed to write to table {table_name}: {e}")
