import unittest

from easyneuron.logging import get_logger
from easyneuron._testutils import log_errors, notRunningInGitHubActions

class TestLogging(unittest.TestCase):
	@log_errors
	def test_get_logger(self):
		if notRunningInGitHubActions():
			logger = get_logger("logs/tests/" + self.__module__ + ".log")
			logger.info("Test complete and working.")
