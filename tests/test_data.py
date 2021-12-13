import unittest
from easyneuron.data import get_random_humans
from easyneuron._testutils import log_errors

class TestDataLoading(unittest.TestCase):

	@log_errors
	def test_random_humans(self):
		get_random_humans() # Try to run it to check for errors.