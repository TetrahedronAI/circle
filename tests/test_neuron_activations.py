import unittest
from easyneuron._testutils.logger import log_errors
from easyneuron.neuron.activations.activations import ReLU, Sigmoid, Tanh


class TestSigmoid(unittest.TestCase):
	@log_errors
	def test_forward(self):
		INPUTS = [-100, 0, 100]
		TARGETS = [0, 0.5, 1]

		for i, j in zip(Sigmoid().forward(INPUTS), TARGETS):
			self.assertAlmostEqual(i, j, 2)

	@log_errors
	def test_backward(self):
		INPUTS = [-100, 0, 100]
		TARGETS = [0, 0.25, 0]

		for i, j in zip(Sigmoid().backward(INPUTS), TARGETS):
			self.assertAlmostEqual(i, j, 2)

class TestTanh(unittest.TestCase):
	@log_errors
	def test_forward(self):
		INPUTS = [-100, -0.2, 0, 0.2, 100]
		TARGETS = [-1, -0.2, 0, 0.2, 1]

		for i, j in zip(Tanh().forward(INPUTS), TARGETS):
			self.assertAlmostEqual(i, j, 2)

	@log_errors
	def test_backward(self):
		INPUTS = [-100, 0, 100]
		TARGETS = [0, 1, 0]

		for i, j in zip(Tanh().backward(INPUTS), TARGETS):
			self.assertAlmostEqual(i, j, 2)

class TestReLU(unittest.TestCase):
	@log_errors
	def test_forward(self):
		INPUTS = [-10, -1, 0, 1, 10]
		TARGETS = [0, 0, 0, 1, 10]

		for i, j in zip(ReLU().forward(INPUTS), TARGETS):
			self.assertEqual(i, j)

	@log_errors
	def test_backward(self):
		INPUTS = [-100, 0, 100]
		TARGETS = [0, 1, 1]

		for i, j in zip(ReLU().backward(INPUTS), TARGETS):
			self.assertEqual(i, j)
