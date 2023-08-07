# Copyright 2023 CircleML GitHub Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Internal error module. Use circleml.core.errors instead."""

import typing as t

from ...log import check_err


class ShapeError(ValueError):
    """The given shape is invalid or mismatched."""


def check_len(x: t.Sized, y: t.Sized) -> None:
    """Check that two objects have the same length.

    Args:
        x (t.Sized): first object
        y (t.Sized): second object

    Raises:
        ShapeError: if the lengths are not equal
    """
    check_err(
        len(x) == len(y), "y_true and y_pred must have the same length", ShapeError
    )
