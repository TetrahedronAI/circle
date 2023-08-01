import unittest
import tests.helpers
import src.circleml.log as log

class TestLogs(unittest.TestCase):
    def test_logggers(self) -> None:
        # check they don't throw erorrs
        log.error("error")
        log.warn("warn")
        log.info("info")
        log.success("success")
        log.debug("debug")

    def test_create_logger(self) -> None:
        l = log.create_logger(lambda _: True, True)
        self.assertTrue(l(True))

        l = log.create_logger(lambda _: True, False)
        self.assertIsNone(l(True))

    def test_checks(self) -> None:
        log.check(True, "error")
        log.check_err(True, "error", Exception)

        with self.assertRaises(SystemExit):
            log.check(False, "error")

        with self.assertRaises(TypeError):
            log.check_err(False, "error", TypeError)