from collections import Counter

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ColumnListTypeFeatureGenerator(BaseEstimator, TransformerMixin):
    """
    This class is a transformer that generates features based on the frequency of IDs in a list-type column.
    It creates binary features for the top N most common IDs, and a count feature for all other IDs.
    """

    def __init__(self, id_col, top_n=30):
        """
        Initialize the transformer with the name of the list-type column and the number of top IDs to consider.

        Parameters:
        id_col (str): The name of the list-type column.
        top_n (int): The number of top IDs to consider.
        """
        self.id_col = id_col
        self.top_n = top_n
        self.top_ids = None

    def fit(self, X, y=None):
        """
        Fit the transformer to the data. This involves computing the top N most common IDs.

        Parameters:
        X (DataFrame): The input data.
        y (Series, optional): The target variable. Not used.

        Returns:
        self
        """
        all_ids = [id for sublist in X[self.id_col] for id in sublist]
        id_counts = Counter(all_ids)
        self.top_ids = [id for id, count in id_counts.most_common(self.top_n)]
        return self

    def transform(self, X):
        """
        Transform the data. This involves creating binary features for the top N IDs and a count feature for all other IDs.

        Parameters:
        X (DataFrame): The input data.

        Returns:
        DataFrame: The transformed data.
        """
        X_transformed = X.copy()
        for id in self.top_ids:
            X_transformed[f"{self.id_col}_{id}"] = X_transformed[self.id_col].apply(
                lambda x: 1 if id in x else 0
            )
        X_transformed[f"{self.id_col}_other"] = X_transformed[self.id_col].apply(
            lambda x: sum(1 for id in x if id not in self.top_ids)
        )
        return X_transformed


class TargetEncoder(BaseEstimator, TransformerMixin):
    """
    Target encoder. Replaces categorical column(s) with the mean target value for
    each category.
    """

    def __init__(self, columns=None):
        """
        Initialize the transformer with the name of the column(s) to encode.

        Parameters:
        columns (list of str): The name of the column(s) to encode. If None, all string columns will be encoded.
        """
        self.columns = columns
        self.encodings = None

    def fit(self, X, y):
        """
        Fit the transformer to the data.

        Parameters:
        X (DataFrame): The input data.
        y (Series): The target variable.

        Returns:
        self
        """
        # If no columns specified, use all string columns
        if self.columns is None:
            self.columns = X.select_dtypes(include=["object"]).columns

        # Compute target means for each category in each column
        self.encodings = {
            column: y.groupby(X[column]).mean() for column in self.columns
        }

        return self

    def transform(self, X):
        """
        Transform the data.

        Parameters:
        X (DataFrame): The input data.

        Returns:
        DataFrame: The data with categorical columns replaced by target means.
        """
        X_transformed = X.copy()

        for column, encoding in self.encodings.items():
            X_transformed[f"target_{column}"] = X_transformed[column].map(encoding)

        return X_transformed


class SetTypeTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer to set the data type of specified columns.
    """

    def __init__(self, columns, dtype):
        """
        Initialize the transformer.

        Parameters:
        columns (list of str): The names of the columns to transform.
        dtype (str): The data type to set.
        """
        self.columns = columns
        self.dtype = dtype

    def fit(self, X, y=None):
        """
        Fit the transformer. This transformer doesn't need to learn anything, so this method just returns self.

        Parameters:
        X (DataFrame): The input data.
        y (Series): The target variable.

        Returns:
        self
        """
        return self

    def transform(self, X):
        """
        Transform the data.

        Parameters:
        X (DataFrame): The input data.

        Returns:
        DataFrame: The data with the data type of the specified columns set to the specified type.
        """
        X_transformed = X.copy()
        X_transformed[self.columns] = X_transformed[self.columns].astype(self.dtype)
        return X_transformed


class InteractionTargetEncoder(BaseEstimator, TransformerMixin):
    """
    Target encoder for interactions. Replaces the interaction of two categorical columns with the mean target value for
    each category.
    """

    def __init__(self, column1, column2):
        """
        Initialize the transformer with the names of the two columns to encode.

        Parameters:
        column1 (str): The name of the first column.
        column2 (str): The name of the second column.
        """
        self.column1 = column1
        self.column2 = column2
        self.encodings = None

    def fit(self, X, y):
        """
        Fit the transformer to the data.

        Parameters:
        X (DataFrame): The input data.
        y (Series): The target variable.

        Returns:
        self
        """
        # Compute target means for each category in the interaction column
        X_interaction = X[self.column1].astype(str) + "_" + X[self.column2].astype(str)
        self.encodings = y.groupby(X_interaction).mean()

        return self

    def transform(self, X):
        """
        Transform the data.

        Parameters:
        X (DataFrame): The input data.

        Returns:
        DataFrame: The data with the interaction column replaced by target means.
        """
        X_transformed = X.copy()

        # Replace the interaction column with the target means
        X_interaction = (
            X_transformed[self.column1].astype(str)
            + "_"
            + X_transformed[self.column2].astype(str)
        )
        X_transformed[
            self.column1 + "_" + self.column2 + "_interaction_target"
        ] = X_interaction.map(self.encodings)
        X_transformed[
            self.column1 + "_" + self.column2 + "_interaction_target_key"
        ] = X_interaction.astype("category")

        return X_transformed


class InteractionTargetEncoderCustom(BaseEstimator, TransformerMixin):
    """
    Target encoder for interactions. Replaces the interaction of two categorical columns with the mean target value for
    each category.
    """

    def __init__(self, column1, column2, ref_id):
        """
        Initialize the transformer with the names of the two columns to encode.

        Parameters:
        column1 (str): The name of the first column.
        column2 (str): The name of the second column.
        """
        self.column1 = column1
        self.column2 = column2
        self.encodings = None

    def fit(self, X, y):
        """
        Fit the transformer to the data.

        Parameters:
        X (DataFrame): The input data.
        y (Series): The target variable.

        Returns:
        self
        """
        # Compute target means for each category in the interaction column
        X_interaction = X[self.column1].astype(str) + "_" + X[self.column2].astype(str)
        self.encodings = y.groupby(X_interaction).mean()

        return self

    def transform(self, X):
        """
        Transform the data.

        Parameters:
        X (DataFrame): The input data.

        Returns:
        DataFrame: The data with the interaction column replaced by target means.
        """
        X_transformed = X.copy()

        # Replace the interaction column with the target means
        X_interaction = (
            X_transformed[self.column1].astype(str)
            + "_"
            + X_transformed[self.column2].astype(str)
        )
        X_transformed[
            self.column1 + "_" + self.column2 + "_interaction_target"
        ] = X_interaction.map(self.encodings)

        return X_transformed


class BucketTargetEncoder(BaseEstimator, TransformerMixin):
    """
    Target encoder for numeric columns. Buckets the numeric columns into bins each containing approximately 5% of the data,
    and replaces each bin with the mean target value for that bin.
    """

    def __init__(self, columns):
        """
        Initialize the transformer with the names of the columns to encode.

        Parameters:
        columns (list of str): The names of the columns to encode.
        """
        self.columns = columns
        self.encodings = {}

    def fit(self, X, y):
        """
        Fit the transformer to the data.

        Parameters:
        X (DataFrame): The input data.
        y (Series): The target variable.

        Returns:
        self
        """
        for column in self.columns:
            X_transformed = X.copy()
            X_transformed[f"{column}_bucket_target"] = pd.cut(
                X_transformed[column], bins=20, duplicates="drop"
            )
            self.encodings[f"{column}_bucket_target"] = y.groupby(
                X_transformed[f"{column}_bucket_target"]
            ).mean()
        return self

    def transform(self, X):
        """
        Transform the data.

        Parameters:
        X (DataFrame): The input data.

        Returns:
        DataFrame: The data with the specified columns replaced by target means.
        """
        X_transformed = X.copy()

        for column in self.columns:
            X_transformed[f"{column}_bucket_target_key"] = pd.cut(
                X_transformed[column], bins=20, duplicates="drop"
            ).astype("category")
            X_transformed[f"{column}_bucket_target"] = X_transformed[
                f"{column}_bucket_target_key"
            ].map(self.encodings[f"{column}_bucket_target"])

        return X_transformed


class DropColumnsTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer to drop specified columns.
    """

    def __init__(self, columns):
        """
        Initialize the transformer.

        Parameters:
        columns (list of str): The names of the columns to drop.
        """
        self.columns = columns

    def fit(self, X, y=None):
        """
        Fit the transformer. This transformer doesn't need to learn anything, so this method just returns self.

        Parameters:
        X (DataFrame): The input data.
        y (Series): The target variable.

        Returns:
        self
        """
        return self

    def transform(self, X):
        """
        Transform the data.

        Parameters:
        X (DataFrame): The input data.

        Returns:
        DataFrame: The data with the specified columns dropped.
        """
        X_transformed = X.copy()
        X_transformed = X_transformed.drop(columns=self.columns)
        return X_transformed


class LogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for col in self.columns:
            X_transformed[col + "_log"] = np.log1p(X_transformed[col])
        return X_transformed
