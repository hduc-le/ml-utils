import logging
import os
from typing import Tuple

import yaml
from google.cloud import storage

logger = logging.getLogger(__name__)


def parse_gcs_path(gcs_path: str) -> Tuple[str, str]:
    """
    Parse the Google Cloud Storage path to get the bucket name and file name.

    Args:
        gcs_path (str): The Google Cloud Storage path.

    Returns:
        Tuple[str, str]: The bucket name and file name.
    """
    bucket_name = gcs_path.split("/")[2]
    gcs_file_name = "/".join(gcs_path.split("/")[3:])
    return bucket_name, gcs_file_name


def upload_from_filename(
    client: storage.Client, local_path: str, gcs_path: str
) -> str:
    """
    Upload a local file to Google Cloud Storage.

    Args:
        client (storage.Client): The Google Cloud Storage client.
        local_path (str): The local file path.
        gcs_path (str): The Google Cloud Storage path.

    Returns:
        str: The Google Cloud Storage path of the uploaded file.
    """
    bucket_name, gcs_file_name = parse_gcs_path(gcs_path)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_file_name)
    try:
        blob.upload_from_filename(local_path)
        logger.info(f"Uploaded file {local_path} to {gcs_path}")
    except Exception as e:
        logger.error(f"Failed to upload file {local_path} to {gcs_path}: {e}")
        raise
    return gcs_path


def download_file_from_gcs(
    client: storage.Client, gcs_path: str, local_path: str
) -> str:
    """
    Download a file from Google Cloud Storage to the local file system.

    Args:
        client (storage.Client): The Google Cloud Storage client.
        gcs_path (str): The Google Cloud Storage path of the file to download.
        local_path (str): The local path where the file will be saved.

    Returns:
        str: The local path of the downloaded file.
    """
    bucket_name, gcs_file_name = parse_gcs_path(gcs_path)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_file_name)
    try:
        blob.download_to_filename(local_path)
        logger.info(f"Downloaded file {gcs_path} to {local_path}")
    except Exception as e:
        logger.error(
            f"Failed to download file {gcs_path} to {local_path}: {e}"
        )
        raise
    return local_path


def read_yaml(file_path):
    """
    Load a YAML file from either a local path or Google Cloud Storage.

    Args:
        file_path (str): Path to the YAML file. It can be a local path or a GCS path.

    Returns:
        dict: Parsed YAML content.
    """
    if file_path.startswith("gs://"):
        # Load from Google Cloud Storage
        client = storage.Client()

        # Parse GCS path
        gcs_path_parts = file_path.replace("gs://", "").split("/", 1)
        bucket_name = gcs_path_parts[0]
        blob_name = gcs_path_parts[1]

        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        yaml_content = blob.download_as_text()
    else:
        # Load from local filesystem
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Local file '{file_path}' does not exist."
            )

        with open(file_path, "r") as f:
            yaml_content = f.read()

    # Parse and return YAML content
    return yaml.safe_load(yaml_content)


def read_sql(file: str) -> str:
    """
    Read a SQL file and return its content.

    Args:
        file (str): The file path.

    Returns:
        str: The content of the SQL file.
    """
    try:
        with open(file, "r") as stream:
            return stream.read()
    except Exception as e:
        logger.error(f"Failed to read SQL file {file}: {e}")
        raise


def gcs_get_last_file_name(
    storage_client: storage.Client, gcs_bucket: str, gcs_path: str
) -> str:
    """
    Get the name of the last file in a Google Cloud Storage directory.

    Args:
        storage_client (storage.Client): The Google Cloud Storage client.
        gcs_bucket (str): The Google Cloud Storage bucket name.
        gcs_path (str): The Google Cloud Storage directory path.

    Returns:
        str: The name of the last file in the directory.
    """
    all_files_in_gcs_dir = []
    for blob in storage_client.list_blobs(gcs_bucket, prefix=gcs_path):
        all_files_in_gcs_dir.append(blob.name)
    last_file = f"gs://{gcs_bucket}/{(all_files_in_gcs_dir)[-1]}"
    logger.info(f"Last file in {gcs_path} is {last_file}")
    return last_file


def gcs_get_subdirs_from_directory(gcs_dir_path: str) -> list:
    if not gcs_dir_path.endswith("/"):
        gcs_dir_path += "/"
    storage_client = storage.Client()
    bucket_name, gcs_dir_path = parse_gcs_path(gcs_dir_path)
    bucket = storage_client.bucket(bucket_name)
    subdirs = []
    for blob in bucket.list_blobs(prefix=gcs_dir_path):
        subdir = blob.name.replace(gcs_dir_path, "").split("/")
        full_subdir = os.path.join(
            "gs://", bucket_name, gcs_dir_path, subdir[0]
        )
        if len(subdir) > 1 and full_subdir not in subdirs:
            subdirs.append(full_subdir)
    return subdirs
