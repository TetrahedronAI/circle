import unittest
from easyneuron.neighbours._classes import _KNNParams
from easyneuron._testutils import log_errors


class TestKNNClasses(unittest.TestCase):
	@log_errors
	def test_knnparams(self):
		test_1 = _KNNParams(1)
		test_2 = _KNNParams(15)
		test_3 = _KNNParams(30)
		test_4 = _KNNParams(126)

		self.assertEqual(test_1.k, 1)
		self.assertEqual(test_2.k, 15)
		self.assertEqual(test_3.k, 30)
		self.assertEqual(test_4.k, 126)
