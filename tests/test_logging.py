import unittest

from sandboxai.logging.logging import get_logger
from sandboxai._testutils import log_errors


class TestLogging(unittest.TestCase):
    @log_errors
    def test_get_logger(self):
        get_logger(f"logs/temp/{self.__module__}.log")