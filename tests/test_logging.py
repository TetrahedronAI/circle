import unittest

from sandboxai.logging.logging import get_logger
from sandboxai._testutils import log_errors, notRunningInGitHubActions


class TestLogging(unittest.TestCase):
    @log_errors
    def test_get_logger(self):
        if notRunningInGitHubActions():
            try:
                logger = get_logger(f"logs/temp/{self.__module__}.log")
                logger.info("Test complete and working.")
            except FileNotFoundError as e:
                raise FileNotFoundError("could not find logs/temp/ folder. Please create it in the workspace. It will be gitignored automatically.") from e
