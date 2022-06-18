import unittest

from sandboxai._testutils import log_errors
from sandboxai.data import make_stairs
from sandboxai.metrics.accuracy.accuracy import accuracy
from sandboxai.neighbours import KNNClassifier


class TestKNNClassifier(unittest.TestCase):
    @log_errors
    def test_params(self):

        for i, j in zip([2, 22], ["euclidean", "manhattan"] * 2):
            test = KNNClassifier(K=i, distance=j)

            self.assertEqual(test.K, i)
            self.assertEqual(test.distance.__name__, f'{j}_distance')

    @log_errors
    def test_raises(self):
        test = KNNClassifier(K=3)

        # Where items in X != items in Y
        self.assertRaises(ValueError, test.fit, [[1, 2, 3] * 2], [5, 6, 7])
        self.assertRaises(ValueError, test.fit, [[1, 2, 3]], [5, 6, 7])

        # Where X has too few dimensions
        self.assertRaises(ValueError, test.fit, [1, 2, 3], [5, 6, 7])
        self.assertRaises(ValueError, test.predict, [1, 2, 3])

    @log_errors
    def test_ordering(self):
        self.assertLess(KNNClassifier(2), KNNClassifier(4))
        self.assertLess(KNNClassifier(20), KNNClassifier(324))

    @log_errors
    def test_warns(self):
        self.assertWarns(FutureWarning, KNNClassifier, K=1)

    @log_errors
    def test_fit(self):
        for i, j, k, l in zip(
            [2, 3, 4, 5, 6, 7],
            [2, 3, 4, 5, 6, 7],
            [2, 3, 4, 5, 6, 7],
            ["euclidean", "manhattan"] * 3,
        ):
            X_train, y_train = make_stairs(i, j)

            test = KNNClassifier(K=k, distance=l)
            test.fit(X_train, y_train)

    def test_predict(self):
        X, y = make_stairs(3, 2, sd=0.1, samples=1000)

        model = KNNClassifier()
        model.fit(X[:900], y[:900])
        preds = model.predict(X[900:])
        self.assertGreaterEqual(accuracy(preds, y[900:]), 0.7)
