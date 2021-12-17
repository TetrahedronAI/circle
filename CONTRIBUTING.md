# Contributing

Please keep to the following contributing guidelines when working on this project.
Before you continue, try running the VS Code task **"New Here"** to get your environment set up before you continue.

## Table of Contents

- [Contributing](#contributing)
	- [Table of Contents](#table-of-contents)
- [Development](#development)
	- [Software Requirements](#software-requirements)
	- [IDE Configuration](#ide-configuration)
		- [Concerning VS Code Users](#concerning-vs-code-users)
		- [Terminal Locations](#terminal-locations)
		- [Important note about the website](#important-note-about-the-website)
		- [Repo Branching](#repo-branching)
	- [Code Style](#code-style)
- [Licensing](#licensing)
	- [Python Message](#python-message)

<br>

---

# Development

## Software Requirements

Aside from an editor (preferably VS Code), Git and Python, that is all you really need for this.

## IDE Configuration

> **First run the `admin/scripts/new_here.py` script to set up pre-commit hooks and dependencies. It would be helpful to rerun this from time to time as new pre-commit hooks are added.** 

*Also, please test before commit.*

### Concerning VS Code Users

If you are a user of Visual Studio Code, you'll find a VS Code configuration with tasks for setup and testing. Please use these, and, if you don't, please do not delete or edit them directly.

There are also a few selected VS Code Extensions that should help, which should come up as workspace recommendations in the VS Code extension tab.

### Terminal Locations

Just a practical note, it helps to have the terminal navigated to the root of the project for running code. It may not otherwise work.

### Important note about the website
If working on the website in the docs folder, please <kbd>CTRL</kbd> + <kbd>K</kbd>, <kbd>CTRL</kbd> + <kbd>O</kbd> (vscode open folder shortcut) into the docs folder.

### Repo Branching
Whether you're working on the main repo or a personal fork, please ensure that you branch it, rather than directly using the master branch, to prevent issues with breaking production. Please name branches all in lower case with dashes with verbose, descriptive names following the following conventions.

| Prefix | Use Case | Example |
| ------ | -------- | ------- |
| `add` | Adding something new | `add-genetic-algorithm-genomes` |
| `improve` | Improving efficiency or design / refactoring | `improve-knn-fit-efficiency` |
| `fix` | Fixing bugs or issues | `fix-security-issue-14` |

## Code Style

Please refer to the following documents for code styles.

| Language | Style File                               |
| -------- | ---------------------------------------- |
| Python   | [admin/python.md](admin/style/python.md) |

<br>

---

# Licensing

As to ensure that it is clear that all files are covered under the correct license (Apache 2.0), please make sure that the files (any new ones you create) begin with the following message. Please fill in the sections marked with the `<>` tags. Use the template below that is appropriate for each language.

## Python Message
Use for Python scripts. There is a `license` snippet that'll work in Python scripts.

```python
# Copyright <YEAR OF FILE CREATION> Neuron-AI GitHub Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
```