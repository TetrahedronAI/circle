import unittest
from easyneuron.neighbours import KNNClassifier
from easyneuron._testutils import log_errors


class TestKNNClasses(unittest.TestCase):
	@log_errors
	def test_knnparams(self):
		test = KNNClassifier(3)

		self.assertEqual(test.K, 3)