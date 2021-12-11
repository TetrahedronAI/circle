import unittest
from easyneuron.neighbours._classes import _KNNParams
from easyneuron._testutils import log_errors


class TestKNNClasses(unittest.TestCase):
	@log_errors
	def test_knnparams(self):
		x1 = _KNNParams(1)
		x2 = _KNNParams(15)
		x3 = _KNNParams(30)
		x4 = _KNNParams(126)

		self.assertEqual(x1.k, 1)
		self.assertEqual(x2.k, 15)
		self.assertEqual(x3.k, 30)
		self.assertEqual(x4.k, 126)
