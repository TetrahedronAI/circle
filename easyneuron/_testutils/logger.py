from typing import Optional
from easyneuron._logging import get_logger

def log_errors(func):
	def wrapper(*args, **kwargs):
		logger = get_logger("logs/" + func.__name__  + ".log")

		try:
			func(*args, **kwargs)
		except Exception as e:
			message = " ".join(str(i) for i in e.args)
			logger.error(f"Function: {func.__name__}- {message}")
			raise e

	return wrapper