import json
import subprocess

import requests

print("Running coverage run...")
subprocess.run("coverage run -m unittest discover")
print("Running coverage report...")
subprocess.run("coverage report -m")
print("Running coverage json...")
subprocess.run("coverage json")

with open("coverage.json", "r") as file:
	cov = json.loads(file.read())["totals"]["percent_covered_display"] + "%"

covInt = int(cov[:-1])
if covInt < 50:
	col = "red"
elif covInt < 75:
	col = "yellow"
else:
	col = "green"

img_data = requests.get(f"https://img.shields.io/static/v1?label=coverage&message={cov}&color={col}").content
with open('admin/social/coverage.svg', 'wb') as handler:
    handler.write(img_data)
