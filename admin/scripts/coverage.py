import json
import subprocess

print("Running coverage run...")
subprocess.run("coverage run -m unittest discover")
print("Running coverage report...")
subprocess.run("coverage report -m")
print("Running coverage json...")
subprocess.run("coverage json")

with open("coverage.json", "r") as file:
	cov = json.loads(file.read())["totals"]["percent_covered_display"] + "%"

with open("admin/social/coverage.svg", "w") as file:
	file.write(f"""<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="100" height="10" role="img"
	aria-label="coverage: planning">
	<img src="https://img.shields.io/static/v1?label=status&message={cov}&color=green" />
</svg>""")
