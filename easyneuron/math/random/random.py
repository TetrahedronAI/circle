import secrets

def secure_random(low: int, high: int) -> int:
	"""Generates a random number from low (inclusive) to high (inclusive) using the secrets module.

	DISCLAIMER: we are not security experts this may not be entirely secure.

	Parameters
	----------
	low : int
		The inclusive lower bound
	high : int
		The inclusive upper bound

	Returns
	-------
	int
		The generated number
	"""
	return secrets.randbelow(high) + low