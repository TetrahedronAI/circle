import os
import unittest

from easyneuron.logging import get_logger
from easyneuron._testutils import log_errors

class TestLogging(unittest.TestCase):
	@log_errors
	def test_get_logger(self):
		if os.environ.get("GITHUB_ACTIONS") not in ["true", "True", "TRUE", True]:
			logger = get_logger("logs/tests/" + self.__module__ + ".log")
			logger.info("Test complete and working.")
