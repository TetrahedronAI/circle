import unittest
from easyneuron.data import get_random_humans, gen_stairs
from easyneuron._testutils import log_errors
from easyneuron.data.gen import gen_stairs

class TestDataLoading(unittest.TestCase):

	@log_errors
	def test_random_humans(self):
		get_random_humans() # just try to run it to check for errors

class TestDataGen(unittest.TestCase):

	@log_errors
	def test_gen_stairs(self):
		x, y = gen_stairs(3, 2) # just try to run it to check for errors