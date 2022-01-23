from io import BufferedWriter
from typing import (Callable, Literal, Sequence, SupportsFloat, SupportsInt,
                    Union)

from numpy import (float16, float32, float64, int0, int8, int16, int32, int64,
                   ndarray)

# Numerical
NumpyFloat = Union[float64, float32, float16]  # Numpy Float types
NumpyInt = Union[int0, int8, int16, int32,
                 int64]  # Numpy Int types
BuiltinNumbers = Union[int, float]  # Builtin number types

# Any numerical, or numpy numerical types
Numerical = Union[BuiltinNumbers, NumpyFloat, NumpyInt, SupportsInt, SupportsFloat]

# External types for set-like objects
ExternalSets = ndarray
BuiltinSets = Union[list, tuple, set]  # Builtin set-like types
SequentialObject = Union[ExternalSets, BuiltinSets]  # All set-like objects

Data = Union[SequentialObject, Numerical] # A Union of sequences that can be used alonside numerical values
X_Data = Sequence[Sequence[Numerical]] # 2D Arrays

# Literal Types
Distance = Literal["euclidean", "manhattan"]  # Distance function names
Loss = Union[Literal["mse", "mae", "mean_squared_error", "mean_absolute_error"], Callable] # Types of losses

# Files
WritableFile = Union[str, BufferedWriter]
