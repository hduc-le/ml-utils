from google.cloud import storage
from google.oauth2 import service_account


class GCSHelper:
    def __init__(self, bucket_name, credentials_path=None):
        """
        Initialize the GCSHelper class.

        :param bucket_name: Name of the Google Cloud Storage bucket.
        :param credentials_path: Path to the service account credentials file (optional).
        """
        self.bucket_name = bucket_name
        if credentials_path:
            credentials = (
                service_account.Credentials.from_service_account_file(
                    credentials_path
                )
            )
            self.client = storage.Client(credentials=credentials)
        else:
            self.client = (
                storage.Client()
            )  # Use default credentials if not provided
        self.bucket = self.client.bucket(self.bucket_name)

    def upload_file(self, file_path, destination_blob_name):
        """
        Uploads a file to the bucket.

        :param file_path: Path of the file to upload.
        :param destination_blob_name: The destination path in the bucket.
        """
        blob = self.bucket.blob(destination_blob_name)
        blob.upload_from_filename(file_path)
        print(f"File {file_path} uploaded to {destination_blob_name}.")

    def upload_from_string(
        self, content, destination_blob_name, content_type="text/plain"
    ):
        """
        Uploads a string to the bucket.

        :param content: The content to upload.
        :param destination_blob_name: The destination path in the bucket.
        :param content_type: The content type of the file (optional).
        """
        blob = self.bucket.blob(destination_blob_name)
        blob.upload_from_string(content, content_type=content_type)
        print(f"Content uploaded to {destination_blob_name}.")

    def download_file(self, source_blob_name, destination_file_name):
        """
        Downloads a file from the bucket.

        :param source_blob_name: The name of the blob in the bucket.
        :param destination_file_name: Path to save the downloaded file.
        """
        blob = self.bucket.blob(source_blob_name)
        blob.download_to_filename(destination_file_name)
        print(
            f"File {source_blob_name} downloaded to {destination_file_name}."
        )

    def download_file_as_text(self, source_blob_name):
        """
        Downloads a file from the bucket and returns its content as text.

        :param source_blob_name: The name of the blob in the bucket.
        :return: The content of the file as text.
        """
        blob = self.bucket.blob(source_blob_name)
        return blob.download_as_text()

    def delete_blob(self, blob_name):
        """
        Deletes a blob from the bucket.

        :param blob_name: The name of the blob to delete.
        """
        blob = self.bucket.blob(blob_name)
        blob.delete()
        print(f"Blob {blob_name} deleted.")

    def delete_folder(self, folder_name):
        """
        Deletes all blobs that are under the specified "folder" prefix.

        :param folder_name: The "folder" prefix to delete.
        """
        blobs = self.bucket.list_blobs(prefix=folder_name)
        for blob in blobs:
            print(f"Deleting {blob.name}...")
            blob.delete()
        print(f"All blobs under {folder_name} have been deleted.")

    def list_blobs(self, prefix=None):
        """
        List all blobs in the bucket, optionally filtered by prefix.

        :param prefix: Filter blobs by prefix (optional).
        :return: List of blob names.
        """
        blobs = self.bucket.list_blobs(prefix=prefix)
        return [blob.name for blob in blobs]

    def file_exists(self, blob_name):
        """
        Check if a file exists in the bucket.

        :param blob_name: The name of the blob to check.
        :return: True if the blob exists, False otherwise.
        """
        blob = self.bucket.blob(blob_name)
        return blob.exists()
