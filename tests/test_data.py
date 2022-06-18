import os
import unittest

from sandboxai._testutils import log_errors
from sandboxai._testutils.gh_actions import notRunningInGitHubActions
from sandboxai.data import make_stairs, load_random_humans


class TestDataLoading(unittest.TestCase):
    @log_errors
    def test_random_humans(self):
        load_random_humans()  # just try to run it to check for errors

        if notRunningInGitHubActions():
            load_random_humans(filename=f"logs/temp/{self.__module__}.csv")
            os.remove(f"logs/temp/{self.__module__}.csv")


class TestDataGen(unittest.TestCase):
    @log_errors
    def test_gen_stairs(self):
        make_stairs(3, 2)  # just try to run it to check for errors