import unittest
import tests.helpers as _
import src.circleml.metrics as metrics

class TestAccuracy(unittest.TestCase):
    def test_all(self):
        y_true = [0, 1, 2, 3, 4]
        y_pred = [0, 1, 2, 3, 4]
        self.assertEqual(metrics.accuracy(y_true, y_pred), 1)
    
    def test_none(self):
        y_true = [0, 1, 2, 3, 4]
        y_pred = [5, 5, 5, 5, 5]
        self.assertEqual(metrics.accuracy(y_true, y_pred), 0)
    
    def test_fraction(self) -> None:
        cases = [
            (
                [0, 1, 2, 3, 4],
                [0, 1, 2, 3, 5],
                0.8
            ),
            (
                [0, 1, 2, 3, 4],
                [0, 2, 3, 4, 5],
                0.2,
            )
        ]

        for y_true, y_pred, expected in cases:
            with self.subTest(y_true=y_true, y_pred=y_pred, expected=expected):
                self.assertEqual(metrics.accuracy(y_true, y_pred), expected)
