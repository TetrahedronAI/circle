import unittest

from easyneuron._testutils import log_errors
from easyneuron.exceptions import DimensionsError
from easyneuron.genetic.genomes import Genome, child_of
from easyneuron.metrics import mean_squared_error
from easyneuron.genetic.optimisers import BasicOptimiser
from numpy.random import randint


class TestGenome(unittest.TestCase):
	@log_errors
	def test_mutate(self):
		TESTS = 10

		noChanges = 0
		for i in [
			[randint(-10, 10) for _ in range(10)]
			for _ in range(TESTS)
		]:
			test = Genome(i)

			old_genome = test.genome.copy()

			new_genome = test.mutate(0.1, magnitude=5).genome

			self.assertTrue(old_genome.shape == new_genome.shape)
			self.assertFalse(old_genome is new_genome, msg="there is an issue with the tests.")

			if (old_genome == new_genome).all(): noChanges += 1

		self.assertLessEqual(noChanges, TESTS * 0.6)

	@log_errors
	def test_child(self):
		test = child_of(
			Genome([[1, 2, 3]]),
			Genome([[1.1, 4, 2.8]])
		)

		self.assertIsInstance(test, Genome)
		self.assertRaises(DimensionsError, child_of, Genome([1, 2, 3]), Genome([1, 2]))

class TestBasicOptimiser(unittest.TestCase):
	@log_errors
	def test_optimiser(self):
		def loss_fn(X, y, genome):
			return mean_squared_error([[1, 2], [3, 4]], genome)

		opt = BasicOptimiser(loss_fn, 100000, genome_shape=(2, 2), lower_bound=1, upper_bound=4)
		solution = opt.optimise([], [], "mae", child_max_loss=0.1, mutation_rate=1, mutation_magnitude=10, fill_value=2)
		print(solution)
