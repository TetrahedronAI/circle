import logging
from typing import Optional

def get_logger(filename: str, log_format: Optional[str] = None):
	if log_format is None:
		log_format = "%(asctime)s \t [%(levelname)s] \t %(message)s"
	
	logging.basicConfig(
		filename=filename,
		level=logging.DEBUG,
		format=log_format
	)
	return logging.getLogger()
