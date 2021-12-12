from typing import Literal, Sequence, SupportsAbs, SupportsFloat, SupportsInt, Union

from numpy import (float16, float32, float64, int0, int8, int16, int32, int64,
                   ndarray)
from pandas import DataFrame, Series

# Numerical
NumpyFloat = Union[float64, float32, float16]  # Numpy Float types
NumpyInt = Union[int0, int8, int16, int32,
                 int64]  # Numpy Int types
BuiltinNumbers = Union[int, float]  # Builtin number types

# Any numerical, or numpy numerical types
Numerical = Union[BuiltinNumbers, NumpyFloat, NumpyInt, SupportsInt, SupportsFloat]
Int = Union[NumpyInt, int]


# External types for set-like objects
ExternalSets = Union[DataFrame, Series, ndarray]
BuiltinSets = Union[list, tuple, set]  # Builtin set-like types
SequentialObject = Union[ExternalSets, BuiltinSets]  # All set-like objects


# A Union of sequences that can be used alonside numerical values
Data = Union[SequentialObject, Numerical]
X_Data = Sequence[Sequence[Numerical]]

# Maths Types
Distance = Literal["euclidean", "manhattan"]  # Distance function names
