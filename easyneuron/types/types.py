from io import BufferedReader, BufferedWriter
from typing import (Any, Iterable, List, Sequence, Set,
                    Sized, SupportsFloat, SupportsInt, Tuple, Union)

from numpy import (float16, float32, float64, int0, int8, int16, int32, int64,
                   ndarray)

# Numerical
NumpyFloat = Union[float64, float32, float16]  # Numpy Float types
NumpyInt = Union[int0, int8, int16, int32, int64]  # Numpy Int types
BuiltinNumbers = Union[int, float]  # Builtin number types

# Any numerical, or numpy numerical types
Numerical = Union[BuiltinNumbers, NumpyFloat, NumpyInt, SupportsInt, SupportsFloat]

# External types for set-like objects
ExternalSets = ndarray
BuiltinSets = Union[List, Tuple, Set]  # Builtin set-like types
SequentialObject = Union[ExternalSets, BuiltinSets]  # All set-like objects

Data = Union[
    SequentialObject, Numerical
]  # A Union of sequences that can be used alonside numerical values

X_Data = Sequence[Sequence[Numerical]]  # 2D Arrays
ArrayLike = Union[SequentialObject, Iterable, Sequence, Sized, Any]


# Files
WritableFile = Union[str, BufferedWriter]
ReadableFile = Union[str, BufferedReader]
