import unittest

from easyneuron._testutils import log_errors
from easyneuron.data import gen_stairs
from easyneuron.metrics.accuracy.accuracy import accuracy
from easyneuron.neighbours import KNNClassifier
from easyneuron.metrics import mean_absolute_error


class TestKNNClassifier(unittest.TestCase):

	@log_errors
	def test_params(self):

		for i, j in zip(
				[2, 22],
				["euclidean", "manhattan"] * 2
		):
			test = KNNClassifier(K=i, distance=j)

			self.assertEqual(test.K, i)
			self.assertEqual(test.distance.__name__, j + "_distance")

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
				["euclidean", "manhattan"] * 3
		):
			X_train, y_train = gen_stairs(i, j)

			test = KNNClassifier(K=k, distance=l)
			test.fit(X_train, y_train)

	def test_predict(self):
		model = KNNClassifier()
		X, y = gen_stairs(3, 3, sd=0.05)

		model.fit(X[:8000], y[:8000])
		preds = model.predict(X[8000:])

		self.assertLessEqual(
			mean_absolute_error(
				preds, y[8000:]
			), 3
		)
		self.assertGreaterEqual(
			accuracy(
				preds, y[8000:]
			), 0.7
		)
