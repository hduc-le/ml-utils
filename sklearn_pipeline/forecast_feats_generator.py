import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from collections import Counter
import pytz

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class InteractionTargetEncoderCustom(BaseEstimator, TransformerMixin):
    """
    Target encoder for interactions. Replaces the interaction of two categorical columns with the mean target value for
    each category.
    """

    def __init__(self, format, column1, ref_id):
        """
        Initialize the transformer with the names of the two columns to encode.

        Parameters:
        column1 (str): The name of the first column.
        column2 (str): The name of the second column.
        """
        self.format = format
        self.column1 = column1
        self.ref_id = ref_id
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
        X_interaction = (
            X[[self.format, self.ref_id]].apply(
                lambda x: self._check_ref_id_format(x[self.ref_id], x[self.format]),
                axis=1,
            )
            + "_"
            + X[self.column1]
        )
        self.encodings = y.groupby(X_interaction).mean()
        return self

    @staticmethod
    def _check_ref_id_format(ref_id, format):
        if format in ["carousel_banner", "half_banner"]:
            if ("HomeScreen" in ref_id) and ("MomoTransactionResult" in ref_id):
                return f"{format}_HomeScreen_MomoTransactionResult"
            elif "HomeScreen" in ref_id:
                return f"{format}_HomeScreen"
            elif "MomoTransactionResult" not in ref_id:
                return f"{format}_MomoTransactionResult"
            else:
                return f"{format}_other"
        elif format in ["thin_banner", "masthead_banner"]:
            if "promotion_hub_2" in ref_id:
                return f"promotion_hub_2_{format}"
            else:
                return f"other_{format}"

        elif format in ["banner"]:
            if "TransferRecent" in ref_id:
                return f"TransferRecent_{format}"
            else:
                return f"other_{format}"

        elif format in ["floating_icon"]:
            if "TransferRecent" in ref_id and "HomeScreen" in ref_id:
                return f"format"
            elif "HomeScreen" in ref_id:
                return f"HomeScreen_{format}"
            elif "TransferRecent" in ref_id:
                return f"TransferRecent_{format}"
            else:
                return f"other_{format}"
        return "other_format"

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
            X_transformed[[self.format, self.ref_id]].apply(
                lambda x: self._check_ref_id_format(x[self.ref_id], x[self.format]),
                axis=1,
            )
            + "_"
            + X_transformed[self.column1]
        )
        X_transformed[
            self.column1 + "_format_ref_interaction_target"
        ] = X_interaction.map(self.encodings)
        X_transformed[self.column1 + "_format_ref_value"] = X_interaction.astype(
            "category"
        )
        return X_transformed


class DateTimeTargetEncoderCustom(BaseEstimator, TransformerMixin):
    """
    Target encoder for interactions. Replaces the interaction of a datetime column and a format column with the mean target value for
    each category.
    """

    def __init__(self, datetime_column, format_column):
        """
        Initialize the transformer with the name of the datetime column and the format column.

        Parameters:
        datetime_column (str): The name of the datetime column.
        format_column (str): The name of the format column.
        """
        self.datetime_column = datetime_column
        self.format_column = format_column
        self.encodings_DOW = None
        self.encodings_DOM = None

    def fit(self, X, y):
        """
        Fit the transformer to the data.

        Parameters:
        X (DataFrame): The input data.
        y (Series): The target variable.

        Returns:
        self
        """
        # Add timezone to the datetime column
        X_transformed = X.copy()
        X_transformed[self.datetime_column] = X_transformed[
            self.datetime_column
        ].dt.tz_localize("Asia/Ho_Chi_Minh")

        # Extract the day of week and day of month from the datetime column
        X_transformed["DOW"] = X_transformed[self.datetime_column].dt.dayofweek
        X_transformed["DOM"] = X_transformed[self.datetime_column].dt.day

        # Compute target means for each category in the interaction column
        self.encodings_DOW = y.groupby(
            [X_transformed["DOW"], X_transformed[self.format_column]]
        ).mean()
        self.encodings_DOM = y.groupby(
            [X_transformed["DOM"], X_transformed[self.format_column]]
        ).mean()

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

        # Add timezone to the datetime column
        X_transformed[self.datetime_column] = X_transformed[
            self.datetime_column
        ].dt.tz_localize("Asia/Ho_Chi_Minh")

        # Extract the day of week and day of month from the datetime column
        X_transformed["DOW"] = X_transformed[self.datetime_column].dt.dayofweek
        X_transformed["DOM"] = X_transformed[self.datetime_column].dt.day

        # Replace the interaction column with the target means
        X_transformed["format_DOW_target"] = (
            X_transformed[["DOW", self.format_column]]
            .apply(tuple, axis=1)
            .map(self.encodings_DOW)
        )
        X_transformed["format_DOM_target"] = (
            X_transformed[["DOM", self.format_column]]
            .apply(tuple, axis=1)
            .map(self.encodings_DOM)
        )

        return X_transformed


class SegmentSizeReplacer(BaseEstimator, TransformerMixin):
    """
    Transformer to replace values in specified columns based on a condition.
    """

    def __init__(self, segment_size, segment_name, new_value):
        """
        Initialize the transformer.

        Parameters:
        columns (list of str): The names of the columns to transform.
        condition (callable): A function that takes a value and returns a boolean.
        new_value (any): The value to replace with when the condition is met.
        """
        self.segment_size = segment_size
        self.segment_name = segment_name
        self.new_value = new_value

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
        DataFrame: The data with the values in the specified columns replaced based on the condition.
        """
        X_transformed = X.copy()
        X_transformed[self.segment_size] = X_transformed[
            [self.segment_size, self.segment_name]
        ].apply(
            lambda x: self.new_value if x[1] == "MASS" or x[0] is None else x[0], axis=1
        )
        return X_transformed


class DaysTargetEncoder(BaseEstimator, TransformerMixin):
    """
    Transformer to calculate the number of days between two dates, bin this number, and replace each bin with the mean target value.
    """

    def __init__(self, start_date_column, end_date_column, new_column):
        """
        Initialize the transformer.

        Parameters:
        start_date_column (str): The name of the start date column.
        end_date_column (str): The name of the end date column.
        new_column (str): The name of the new column to store the number of days.
        """
        self.start_date_column = start_date_column
        self.end_date_column = end_date_column
        self.new_column = new_column
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
        # Calculate the number of days between the start and end dates
        X_transformed = X.copy()
        X_transformed[self.start_date_column] = pd.to_datetime(
            X_transformed[self.start_date_column]
        )
        X_transformed[self.end_date_column] = pd.to_datetime(
            X_transformed[self.end_date_column]
        )

        # Ensure both datetime columns are timezone aware
        if X_transformed[self.start_date_column].dt.tz is None:
            X_transformed[self.start_date_column] = X_transformed[
                self.start_date_column
            ].dt.tz_localize("Asia/Ho_Chi_Minh")
        if X_transformed[self.end_date_column].dt.tz is None:
            X_transformed[self.end_date_column] = X_transformed[
                self.end_date_column
            ].dt.tz_localize("Asia/Ho_Chi_Minh")

        X_transformed[self.new_column] = (
            X_transformed[self.end_date_column] - X_transformed[self.start_date_column]
        ).dt.days
        # Bin the number of days into 20 categories
        X_transformed[self.new_column] = pd.cut(
            X_transformed[self.new_column], bins=20, labels=False
        )
        # Compute target means for each bin
        self.encodings = y.groupby(X_transformed[self.new_column]).mean()

        return self

    def transform(self, X):
        """
        Transform the data.

        Parameters:
        X (DataFrame): The input data.

        Returns:
        DataFrame: The data with the number of days replaced by target means.
        """
        X_transformed = X.copy()
        X_transformed[self.start_date_column] = pd.to_datetime(
            X_transformed[self.start_date_column]
        )
        X_transformed[self.end_date_column] = pd.to_datetime(
            X_transformed[self.end_date_column]
        )

        # Ensure both datetime columns are timezone aware
        if X_transformed[self.start_date_column].dt.tz is None:
            X_transformed[self.start_date_column] = X_transformed[
                self.start_date_column
            ].dt.tz_localize("Asia/Ho_Chi_Minh")
        if X_transformed[self.end_date_column].dt.tz is None:
            X_transformed[self.end_date_column] = X_transformed[
                self.end_date_column
            ].dt.tz_localize("Asia/Ho_Chi_Minh")

        # Calculate the number of days between the start and end dates
        X_transformed[self.new_column] = (
            X_transformed[self.end_date_column] - X_transformed[self.start_date_column]
        ).dt.days

        # Bin the number of days into 20 categories
        X_transformed[self.new_column] = pd.cut(
            X_transformed[self.new_column], bins=20, labels=False
        )

        # Replace the bins with the target means
        X_transformed[self.new_column] = X_transformed[self.new_column].map(
            self.encodings
        )

        return X_transformed


class BudgetDaysTargetEncoder(BaseEstimator, TransformerMixin):
    """
    Transformer to calculate the number of days between two dates, bin this number, and replace each bin with the mean target value.
    """

    def __init__(self, budget, start_date_column, end_date_column, new_column):
        """
        Initialize the transformer.

        Parameters:
        start_date_column (str): The name of the start date column.
        end_date_column (str): The name of the end date column.
        new_column (str): The name of the new column to store the number of days.
        """
        self.budget = budget
        self.start_date_column = start_date_column
        self.end_date_column = end_date_column
        self.new_column = new_column
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
        # Calculate the number of days between the start and end dates
        X_transformed = X.copy()
        X_transformed[self.start_date_column] = pd.to_datetime(
            X_transformed[self.start_date_column]
        )
        X_transformed[self.end_date_column] = pd.to_datetime(
            X_transformed[self.end_date_column]
        )

        X_transformed[self.budget + "_bin"] = pd.cut(
            X_transformed[self.budget], bins=20, labels=False
        )

        # Ensure both datetime columns are timezone aware
        if X_transformed[self.start_date_column].dt.tz is None:
            X_transformed[self.start_date_column] = X_transformed[
                self.start_date_column
            ].dt.tz_localize("Asia/Ho_Chi_Minh")
        if X_transformed[self.end_date_column].dt.tz is None:
            X_transformed[self.end_date_column] = X_transformed[
                self.end_date_column
            ].dt.tz_localize("Asia/Ho_Chi_Minh")

        X_transformed[self.budget + "_" + "num_days_rate"] = X_transformed[
            self.budget
        ] / (
            (
                X_transformed[self.end_date_column]
                - X_transformed[self.start_date_column]
            ).dt.days
            + 1
        )

        # Replace positive infinity values with NaN
        X_transformed.loc[
            X_transformed[self.budget + "_" + "num_days_rate"] == np.inf,
            self.budget + "_" + "num_days_rate",
        ] = np.nan
        # Replace negative infinity values with NaN
        X_transformed.loc[
            X_transformed[self.budget + "_" + "num_days_rate"] == -np.inf,
            self.budget + "_" + "num_days_rate",
        ] = np.nan
        # Now you can use pd.cut function
        X_transformed[self.budget + "_" + "num_days_rate_bin"] = pd.cut(
            X_transformed[self.budget + "_" + "num_days_rate"], bins=20, labels=False
        )
        # Bin the number of days into 20 categories
        X_transformed[self.budget + "_" + "num_days_rate_bin"] = pd.cut(
            X_transformed[self.budget + "_" + "num_days_rate"].fillna(0),
            bins=20,
            labels=False,
        )
        # Compute target means for each bin
        self.encodings = y.groupby(
            X_transformed[self.budget + "_" + "num_days_rate_bin"]
        ).mean()

        return self

    def transform(self, X):
        """
        Transform the data.

        Parameters:
        X (DataFrame): The input data.

        Returns:
        DataFrame: The data with the number of days replaced by target means.
        """
        X_transformed = X.copy()
        X_transformed[self.start_date_column] = pd.to_datetime(
            X_transformed[self.start_date_column]
        )
        X_transformed[self.end_date_column] = pd.to_datetime(
            X_transformed[self.end_date_column]
        )

        # Ensure both datetime columns are timezone aware
        if X_transformed[self.start_date_column].dt.tz is None:
            X_transformed[self.start_date_column] = X_transformed[
                self.start_date_column
            ].dt.tz_localize("Asia/Ho_Chi_Minh")
        if X_transformed[self.end_date_column].dt.tz is None:
            X_transformed[self.end_date_column] = X_transformed[
                self.end_date_column
            ].dt.tz_localize("Asia/Ho_Chi_Minh")

        # Calculate the number of days between the start and end dates
        X_transformed[self.budget + "_" + "num_days_rate"] = (
            X_transformed[self.budget]
            / (
                X_transformed[self.end_date_column]
                - X_transformed[self.start_date_column]
            ).dt.days
        )
        # Replace positive infinity values with NaN
        X_transformed.loc[
            X_transformed[self.budget + "_" + "num_days_rate"] == np.inf,
            self.budget + "_" + "num_days_rate",
        ] = np.nan
        # Replace negative infinity values with NaN
        X_transformed.loc[
            X_transformed[self.budget + "_" + "num_days_rate"] == -np.inf,
            self.budget + "_" + "num_days_rate",
        ] = np.nan
        # Bin the number of days into 20 categories
        X_transformed[self.budget + "_" + "num_days_rate_bin"] = pd.cut(
            X_transformed[self.budget + "_" + "num_days_rate"], bins=20, labels=False
        )

        # Replace the bins with the target means
        X_transformed[self.budget + "_" + "num_days_rate_bin"] = X_transformed[
            self.budget + "_" + "num_days_rate_bin"
        ].map(self.encodings)

        return X_transformed


class ListStringTargetEncoder(BaseEstimator, TransformerMixin):
    """
    Target encoder for a list of string columns. Replaces each category in the specified columns with the mean target value.
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
        # Compute target means for each category in each column
        for column in self.columns:
            self.encodings[column] = y.groupby(X[column]).mean()
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
            X_transformed["targeting_" + column] = X_transformed[column].map(
                self.encodings[column]
            )
        return X_transformed


class CleanStringTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer to clean string values by removing numeric characters and trailing underscores.
    """

    def __init__(self, columns):
        """
        Initialize the transformer with the names of the columns to clean.

        Parameters:
        columns (list of str): The names of the columns to clean.
        """
        self.columns = columns

    @staticmethod
    def clean_string(s):
        """
        Function to remove numeric values and trailing underscores from a string.

        Parameters:
        s (str): The input string.

        Returns:
        str: The cleaned string.
        """
        # Remove numeric values
        s = re.sub(r"\d+", "", s)
        # Remove trailing underscores
        s = re.sub(r"_+$", "", s)
        return s

    def fit(self, X, y=None):
        """
        Fit the transformer to the data.

        Parameters:
        X (DataFrame): The input data.
        y (Series, optional): The target variable. Default is None.

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
        DataFrame: The data with cleaned string values in the specified columns.
        """
        X_transformed = X.copy()
        for column in self.columns:
            X_transformed[column] = X_transformed[column].apply(self.clean_string)
        return X_transformed
