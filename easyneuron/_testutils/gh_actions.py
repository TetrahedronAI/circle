from os import environ

def notRunningInGitHubActions() -> bool:
	"""Ensures that the process is not running in a GitHub action

	Returns
	-------
	bool
		True if running locally, False if in a GitHub action
	"""
	return environ.get("GITHUB_ACTIONS") not in ["true", "True", "TRUE", True]