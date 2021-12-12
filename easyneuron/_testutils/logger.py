from easyneuron._logging import get_logger


def log_errors(func):
    def wrapper(*args, **kwargs):
        logger = get_logger("logs/" + func.__module__ + ".log")

        try:
            func(*args, **kwargs)
            logger.info(f"{func.__qualname__} - Test Success")
        except Exception as e:
            message = " ".join(str(i) for i in e.args)
            logger.error(
                f"\"{str(type(e))[8:-2]}\" @ {func.__qualname__} - {message}")
            raise e

    return wrapper
