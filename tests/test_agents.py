import unittest

from sandboxai.agents.envs import Environment
from sandboxai.agents.envs.examples import SimpleLateralMover
from sandboxai.agents.qlearn import QTable

# Environments
class TestEnvironment(unittest.TestCase):
    def test_base_functionality(self):
        self.assertRaises(Exception, Environment)

class TestSimpleLateralMover(unittest.TestCase):
    def test_env_tester_lateralmover(self):
        env = SimpleLateralMover()
        env.reset()
        env.get_actions()
        env.get_obs()
        env.step(1)

        self.assertEqual(env.get_obs_shape(), (100,))
        self.assertEqual(env.get_all_actions(), [-1, 0, 1])

class TestQTable(unittest.TestCase):
    def test_qtable_init(self):
        qtable = QTable(3)
        self.assertEqual(qtable.actions, [0, 1, 2])
        self.assertEqual(qtable.states, [])

    def test_qtable_add_state(self):
        qtable = QTable(3)
        qtable.add_state(0)
        self.assertEqual(qtable.states, [0])
        self.assertEqual(qtable.table.shape, (1, 3))
        qtable.add_state(1)
        self.assertEqual(qtable.states, [0, 1])
        self.assertEqual(qtable.table.shape, (2, 3))

    def test_qtable_update(self):
        qtable = QTable(3)
        qtable.add_state(0)
        qtable.add_state(1)
        qtable.update(0, 0, 1)
        self.assertEqual(qtable.table[0][0], 1)
        qtable.update(1, 1, 1)
        self.assertEqual(qtable.table[1][1], 1)