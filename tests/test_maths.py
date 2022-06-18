import unittest

from sandboxai._testutils import log_errors
from sandboxai.math import euclidean_distance, manhattan_distance


class TestDistance(unittest.TestCase):
	@log_errors
	def test_euclidean(self):
		self.assertEqual(euclidean_distance([1, 2, 3], [1, 2, 3]), 0)
		self.assertEqual(euclidean_distance([1], [2]), 1)
		self.assertAlmostEqual(euclidean_distance([[1, 2, 3]], [[2, 3, 4]]), 1.732, 3)

		self.assertWarns(UserWarning, euclidean_distance, [1, 2, 3], [4, 5])

	@log_errors
	def test_manhattan(self):
		self.assertEqual(manhattan_distance([1, 2, 3], [1, 2, 3]), 0)
		self.assertEqual(manhattan_distance([1], [2]), 1)
		self.assertEqual(manhattan_distance([[1, 2, 3]], [[2, 3, 4]]), 3)

		self.assertWarns(UserWarning, manhattan_distance, [1, 2, 3], [4, 5])
