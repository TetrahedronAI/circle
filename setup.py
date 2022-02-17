import codecs
import json
import os

from setuptools import find_packages, setup

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

with open("info.json", "r") as file:
    info_json = json.loads(file.read())

# Setting up
setup(
    name=info_json.get("name"),
    version=info_json.get("version"),
    author=info_json.get("author"),
    author_email=info_json.get("email"),
    description=info_json.get("description"),
    long_description_content_type="text/markdown",
    long_description=info_json.get("long_description"),
    py_version=info_json.get("py_version"),
    packages=find_packages(),
    keywords=info_json.get("keywords"),
    classifiers=info_json.get("classifiers"),
)
