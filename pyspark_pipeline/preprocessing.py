import json
import os
from typing import List

from pyspark.ml import Pipeline, PipelineModel, Transformer
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from ..storage_helper import GCSHelper
from ..transformer.imputer import ConstantImputer


class PySparkPreprocessingPipeline:
    def __init__(
        self,
        feature_names: List[str],
        feature_column: str = "features",
        numerical_default: int = -9999,
        categorical_default: str = "unknown",
    ):
        """
        Initializes the preprocessing pipeline which imputes missing values,
        encodes categorical columns, and assembles feature vectors.

        Args:
            feature_names (List[str]): List of feature column names.
            feature_column (str, optional): Name of the output feature vector column. Defaults to "features".
            numerical_default (int, optional): Default value for missing numerical values. Defaults to -9999.
            categorical_default (str, optional): Default value for missing categorical values. Defaults to "unknown".
        """
        self.feature_names = feature_names
        self.feature_column = feature_column
        self.numerical_default = numerical_default
        self.categorical_default = categorical_default
        self.pipeline_model = None  # Will hold the fitted pipeline model

        # Initialize metadata as a dictionary
        self.metadata = {
            "categorical_imputer_index": None,
            "string_indexer_index": None,
            "numerical_imputer_index": None,
            "feature_names": feature_names,
            "feature_column": feature_column,
            "numerical_default": numerical_default,
            "categorical_default": categorical_default,
            "categorical_mappings": {},
        }

    def _get_stages(self, spark_df: DataFrame) -> List[Transformer]:
        """Creates and returns the stages of the pipeline based on the input DataFrame."""
        categorical_columns = [
            item[0]
            for item in spark_df.dtypes
            if item[0] in self.feature_names and item[1].startswith("string")
        ]
        numerical_columns = [
            item[0]
            for item in spark_df.dtypes
            if item[0] in self.feature_names
            and item[0] not in categorical_columns
        ]
        categorical_columns_output = [
            f"{col}_indexed" for col in categorical_columns
        ]

        stages = []

        # Categorical imputation and indexing
        if categorical_columns:
            categorical_imputer = ConstantImputer(
                inputCols=categorical_columns,
                defaultValue=self.categorical_default,
            )
            string_indexer = StringIndexer(
                inputCols=categorical_columns,
                outputCols=categorical_columns_output,
                handleInvalid="keep",
            )
            self.metadata["categorical_imputer_index"] = len(stages)
            stages.append(categorical_imputer)
            self.metadata["string_indexer_index"] = len(stages)
            stages.append(string_indexer)

        # Numerical imputation
        if numerical_columns:
            numerical_imputer = ConstantImputer(
                inputCols=numerical_columns,
                defaultValue=self.numerical_default,
            )
            self.metadata["numerical_imputer_index"] = len(stages)
            stages.append(numerical_imputer)

        # VectorAssembler for feature vector
        vector_assembler_columns = [
            f"{col}_indexed" if col in categorical_columns else col
            for col in self.feature_names
        ]
        vector_assembler = VectorAssembler(
            inputCols=vector_assembler_columns, outputCol=self.feature_column
        )
        stages.append(vector_assembler)

        return stages

    def _extract_categorical_mappings(self):
        """Extracts the mappings of categorical columns to their indexed values from the fitted StringIndexer."""
        if self.metadata["string_indexer_index"] is not None:
            string_indexer = self.pipeline_model.stages[
                self.metadata["string_indexer_index"]
            ]
            input_cols = string_indexer.getInputCols()
            labels_array = string_indexer.labelsArray

            # Save the mappings for each categorical column
            for idx, col in enumerate(input_cols):
                self.metadata["categorical_mappings"][col] = {
                    k: v
                    if k != self.categorical_default
                    else self.numerical_default
                    for v, k in enumerate(labels_array[idx])
                }

    def fit(self, spark_df: DataFrame):
        """
        Fits the preprocessing pipeline to the input DataFrame and stores the pipeline model.

        Args:
            spark_df (DataFrame): Input Spark DataFrame.

        Returns:
            self: Fitted DataPreprocessingPipeline instance.
        """
        stages = self._get_stages(spark_df)
        pipeline = Pipeline(stages=stages)

        # Fit the pipeline on the input DataFrame
        self.pipeline_model = pipeline.fit(spark_df)

        # Extract and store categorical mappings
        self._extract_categorical_mappings()

        return self

    def transform(self, spark_df: DataFrame) -> DataFrame:
        """
        Transforms the input DataFrame using the fitted pipeline model.

        Args:
            spark_df (DataFrame): Input Spark DataFrame.

        Returns:
            DataFrame: Transformed DataFrame with processed features.
        """
        if not self.pipeline_model:
            raise ValueError(
                "Pipeline model is not fitted. Call 'fit' before 'transform'."
            )

        # Store the original column order (excluding the feature column)
        original_columns = spark_df.columns

        # Apply the transformation using the fitted pipeline
        spark_df = self.pipeline_model.transform(spark_df)

        # If categorical imputation and StringIndexer are present
        if self.metadata["string_indexer_index"] is not None:
            string_indexer = self.pipeline_model.stages[
                self.metadata["string_indexer_index"]
            ]
            categorical_input_columns = string_indexer.getInputCols()
            categorical_output_columns = string_indexer.getOutputCols()

            # Replace unknown placeholders with default numeric value for categorical columns
            for cname_idx, cname_output in enumerate(
                categorical_output_columns
            ):
                labels_array = string_indexer.labelsArray[cname_idx]
                if self.categorical_default in labels_array:
                    categorical_missing_index = labels_array.index(
                        self.categorical_default
                    )
                    spark_df = spark_df.withColumn(
                        cname_output,
                        F.when(
                            F.col(cname_output) == categorical_missing_index,
                            F.lit(self.numerical_default),
                        ).otherwise(F.col(cname_output)),
                    )

            # Drop original categorical columns and rename indexed columns back to their original names
            spark_df = spark_df.drop(*categorical_input_columns)
            for new_name, old_name in zip(
                categorical_output_columns, categorical_input_columns
            ):
                spark_df = spark_df.withColumnRenamed(new_name, old_name)

        # Vectorize the feature column as an array
        spark_df = spark_df.withColumn(
            self.feature_column, vector_to_array(F.col(self.feature_column))
        )

        # Reorder the columns to match the original column order, appending the feature column at the end
        reordered_columns = original_columns + [self.feature_column]
        spark_df = spark_df.select(*reordered_columns)

        return spark_df

    def fit_transform(self, spark_df: DataFrame) -> DataFrame:
        """
        Fits the pipeline to the input DataFrame and then transforms it.

        Args:
            spark_df (DataFrame): Input Spark DataFrame.

        Returns:
            DataFrame: Transformed DataFrame with processed features.
        """
        return self.fit(spark_df).transform(spark_df)

    def save(self, save_path: str):
        """
        Saves the fitted pipeline model and metadata (indexes and categorical mappings) to the specified path.

        Args:
            save_path (str): Path to save the pipeline model and metadata.
        """
        if self.pipeline_model:
            # Save the pipeline model
            self.pipeline_model.write().overwrite().save(
                os.path.join(save_path, "pipeline_model")
            )

            # Save the metadata to either local storage or GCS
            if save_path.startswith("gs://"):
                self._save_metadata_to_gcs(save_path)
            else:
                # Save metadata locally
                with open(
                    os.path.join(save_path, "metadata.json"), "w"
                ) as metadata_file:
                    json.dump(self.metadata, metadata_file)
        else:
            raise ValueError(
                "Pipeline model is not fitted. Cannot save an unfitted pipeline."
            )

    @classmethod
    def load(cls, load_path: str):
        """
        Loads a pipeline model and metadata from the specified path.

        Args:
            load_path (str): Path to load the pipeline model and metadata from.

        Returns:
            DataPreprocessingPipeline: Instance with the loaded pipeline model and metadata.
        """
        if load_path.startswith("gs://"):
            # Load metadata from GCS
            metadata = cls._load_metadata_from_gcs(load_path)
        else:
            # Load metadata from local storage
            metadata_path = os.path.join(load_path, "metadata.json")
            if not os.path.exists(metadata_path):
                raise ValueError(f"Metadata file not found at {metadata_path}")

            with open(metadata_path, "r") as metadata_file:
                metadata = json.load(metadata_file)

        # Create an instance of the class
        instance = cls(
            feature_names=metadata["feature_names"],
            feature_column=metadata["feature_column"],
            numerical_default=metadata["numerical_default"],
            categorical_default=metadata["categorical_default"],
        )

        # Load the pipeline model
        instance.pipeline_model = PipelineModel.load(
            os.path.join(load_path, "pipeline_model")
        )

        # Load the stored metadata (e.g., stage indexes and mappings)
        instance.metadata = metadata

        return instance

    def _save_metadata_to_gcs(self, gcs_path: str):
        """
        Helper method to save metadata to a GCS bucket.

        Args:
            gcs_path (str): The GCS path where metadata should be saved (should be in the format gs://bucket_name/path_to_metadata).
        """
        # Extract the bucket name and blob path
        if not gcs_path.startswith("gs://"):
            raise ValueError(
                "Invalid GCS path. Path should start with 'gs://'."
            )

        gcs_path = gcs_path[5:]  # Strip 'gs://' prefix
        bucket_name, *blob_path_parts = gcs_path.split("/")
        blob_path = "/".join(blob_path_parts) + "/metadata.json"

        # Initialize the GCS client and create a blob to upload the metadata file
        gcs_helper = GCSHelper(bucket_name=bucket_name)

        # Convert the metadata dictionary to a JSON string
        metadata_json_str = json.dumps(self.metadata)

        # Upload the JSON string to GCS
        gcs_helper.upload_from_string(
            metadata_json_str, blob_path, content_type="application/json"
        )

    @staticmethod
    def _load_metadata_from_gcs(gcs_path: str) -> dict:
        """
        Helper method to load metadata from a GCS bucket.

        Args:
            gcs_path (str): The GCS path to the metadata (should be in the format gs://bucket_name/path_to_metadata).

        Returns:
            dict: The metadata loaded from GCS as a dictionary.
        """
        # Extract the bucket name and blob path
        if not gcs_path.startswith("gs://"):
            raise ValueError(
                "Invalid GCS path. Path should start with 'gs://'."
            )

        gcs_path = gcs_path[5:]  # Strip 'gs://' prefix
        bucket_name, *blob_path_parts = gcs_path.split("/")
        blob_path = "/".join(blob_path_parts) + "/metadata.json"

        # Initialize the GCS client and get the metadata file
        gcs_helper = GCSHelper(bucket_name=bucket_name)

        # Download the metadata JSON content as a string
        metadata_json_str = gcs_helper.download_file_as_text(blob_path)

        # Parse the JSON string into a dictionary
        metadata = json.loads(metadata_json_str)

        return metadata
