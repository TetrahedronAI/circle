print("You are new here! Please see CONTRIBUTING.md for contribution guidelines.\n")

import subprocess
subprocess.run("pip install -r requirements.txt")
subprocess.run("pip install -U coverage")

with open("admin/scripts/pre-commit", "r") as file:
	with open(".git/hooks/pre-commit", "w") as hook:
		hook.write(file.read())