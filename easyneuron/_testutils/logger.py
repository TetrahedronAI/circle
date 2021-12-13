import os

from easyneuron.logging import get_logger


def log_errors(func):
    def wrapper(*args, **kwargs):
        if os.environ.get("GITHUB_ACTIONS") not in ["true", "True", "TRUE", True]:
            logger = get_logger("logs/" + func.__module__ + ".log")

            try:
                func(*args, **kwargs)
                logger.info(f"{func.__qualname__} - Test Success")
            except Exception as e:
                message = " ".join(str(i) for i in e.args).replace("\n", " ").replace("  ", "")
                logger.error(
                    f"\"{str(type(e))[8:-2]}\" @ {func.__qualname__} - {message}")
                raise e
        else:
            func(*args, **kwargs)

    return wrapper
