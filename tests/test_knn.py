import unittest
import tests.helpers
import src.circleml.knn as knn

class TestKNNClassifier(unittest.TestCase):
    def test_tie(self) -> None:
        model = knn.KNNCla(2)
        model.fit([[1, 2], [3, 4]], [0, 1])
        p = model.predict([[3, 4]])
        self.assertEqual(p, 1)

    def test_base(self) -> None:
        model = knn.KNNCla()
        model.fit([[1, 2], [3, 4], [5, 6]], [0, 1, 1])
        p = model.predict([[3, 4]])
        self.assertEqual(p, 1)

    def test_rangecheck(self) -> None:
        model = knn.KNNCla()
        with self.assertRaises(SystemExit):
            model.fit([[1]], [0, 1])