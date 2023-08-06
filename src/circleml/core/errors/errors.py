from ...log import check_err

class ShapeError(ValueError):
    """The given shape is invalid or mismatched."""

check_len = lambda x, y: check_err(len(x) == len(y), "y_true and y_pred must have the same length", ShapeError)
