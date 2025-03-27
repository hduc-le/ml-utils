from typing import List, Union

from pyspark import keyword_only
from pyspark.ml import Transformer
from pyspark.ml.param.shared import Param
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql import DataFrame


class NoOpTransformer(
    Transformer, DefaultParamsReadable, DefaultParamsWritable
):
    """
    A no-op transformer that does nothing to the DataFrame by default.
    Provides methods for subclasses to specify inputCols and defaultValue.
    """

    def __init__(self):
        super(NoOpTransformer, self).__init__()
        # These parameters are not used in NoOpTransformer but allow subclasses to implement them
        self.inputCols = Param(self, "inputCols", "List of columns to process")
        self.defaultValue = Param(
            self,
            "defaultValue",
            "Default value for imputation or other operations",
        )

    def getInputCols(self):
        """Returns the input columns. None for NoOpTransformer by default."""
        return None

    def getDefaultValue(self):
        """Returns the default value. None for NoOpTransformer by default."""
        return None

    def _transform(self, dataset: DataFrame) -> DataFrame:
        """No operation transformation that simply returns the dataset unchanged."""
        return dataset


class ConstantImputer(NoOpTransformer):
    """
    Custom transformer for imputing missing numerical values with a constant value.

    Args:
        defaultValue (int or float): Value to replace missing numerical values. Default is -9999.
    """

    @keyword_only
    def __init__(
        self, inputCols: List[str] = None, defaultValue: Union[str, int] = None
    ):
        super(ConstantImputer, self).__init__()
        self._setDefault(defaultValue=defaultValue)
        self._set(**self._input_kwargs)

    def getInputCols(self):
        return self.getOrDefault(self.inputCols)

    def getDefaultValue(self):
        return self.getOrDefault(self.defaultValue)

    def _transform(self, dataset: DataFrame) -> DataFrame:
        input_cols = self.getInputCols()
        default_value = self.getDefaultValue()

        # If inputCols or defaultValue is None, perform no operation (fall back to NoOpTransformer)
        if not input_cols or default_value is None:
            return super()._transform(dataset)  # No-op behavior

        dataset = dataset.fillna(default_value, subset=input_cols)
        return dataset
