import unittest

from easyneuron.agents.envs import Environment
from easyneuron.agents.envs.examples import SimpleLateralMover


class TestEnvironments(unittest.TestCase):

    def test_env_tester_lateralmover(self):
        env = SimpleLateralMover()
        env.reset()
        env.get_actions()
        env.get_obs()
        env.step(1)

        self.assertEqual(env.get_obs_shape(), (100,))
        self.assertEqual(env.get_all_actions(), [-1, 0, 1])
