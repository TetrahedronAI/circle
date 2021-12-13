import unittest
from easyneuron.neighbours import KNNClassifier
from easyneuron.neighbours._classes import _KNN
from easyneuron._testutils import log_errors


class TestKNNClasses(unittest.TestCase):

	@log_errors
	def test_params(self):
		for i, j in zip(
			[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
			["euclidean", "manhattan"] * 5
			):
			test = KNNClassifier(K=i, distance=j)

			self.assertEqual(test.K, i)
			self.assertEqual(test.distance.__name__, j + "_distance")
