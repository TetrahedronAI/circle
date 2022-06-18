import unittest

from sandboxai._testutils import log_errors
from sandboxai.exceptions import DimensionsError
from sandboxai.metrics import accuracy, mean_absolute_error, mean_squared_error


class TestMeanErrors(unittest.TestCase):
    @log_errors
    def test_mse(self):
        self.assertEqual(mean_squared_error([1, 2, 3], [2, 3, 4]), 1)
        self.assertEqual(mean_squared_error([1, 2, 3], [3, 4, 5]), 4)
        self.assertEqual(
            mean_squared_error([[1, 2, 3], [0, -1, -2]], [[2, 3, 4], [-1, -2, -3]]), 1
        )

    @log_errors
    def test_mae(self):
        self.assertEqual(mean_absolute_error([1, 2, 3], [2, 3, 4]), 1)
        self.assertEqual(mean_absolute_error([1, 2, 3], [3, 4, 5]), 2)
        self.assertEqual(
            mean_absolute_error([[1, 2, 3], [0, -1, -2]], [[2, 3, 4], [-1, -2, -3]]), 1
        )


class TestAccuracy(unittest.TestCase):
    @log_errors
    def test_accuracy(self):
        self.assertRaises(DimensionsError, accuracy, [1, 2, 3], [1, 2])

        self.assertEqual(accuracy([1, 2, 3], [1, 2, 3]), 1)
        self.assertEqual(accuracy([1, 2, 3, 4], [1, 2, 3, 5]), 0.75)
        self.assertEqual(accuracy([1, 2, 3, 4], [1, 2, 4, 5]), 0.5)
        self.assertEqual(accuracy([1, 2, 3, 4], [1, 3, 4, 5]), 0.25)
        self.assertEqual(accuracy([1, 2, 3, 4], [0, 3, 4, 5]), 0)
