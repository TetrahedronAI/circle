import unittest

from easyneuron._testutils import log_errors
from easyneuron.neuron.layers.dense import Dense


class TestDense(unittest.TestCase):
    @log_errors
    def test_forward(self):
        layer = Dense(3, 2)
        layer.forward([1, 2])
