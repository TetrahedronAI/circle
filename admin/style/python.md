# Python Coding Style

## Testing
We use the built in **unittest** module for testing. Please use it too, since it appears as part of the test explorer in the sidebar.

Also, lease use the provided testing utils for testing, as seen below...

The decorator that logs the files is as below. **(This requires you to already have a folder called *`logs`*)**.

```python
from easyneuron._testutils import log_errors
```

Then, for every **unittest** test, simply decorate it with `@log_errors`.

## Code Style

<ol>
<li>Ensure that code is documented according to the numpy style. This can be done with the <b>Python Auto Docstrings</b> vscode extension and setting your default style to Numpy in vscode settings. (This should be done automatically if you have the extension and the vscode settings that come with this repo.

<br>
Alternatively, use the template below...
</li>

	"""[summary]
	Parameters
	----------
	<PARAM NAME> : <TYPE OF PARAM>
		[description (what it is for)]

	Returns
	-------
	<TYPE OF RETURN>
		[description (what it is)]

	Raises
	------
	<ERROR TYPE>
		[description (why it's raised)]
	"""

<br>

<li>If errors in the function are raised (on purpose - by guard clauses, not bugs), be sure to put a solution as stated in the [docs_errors.md], to specify a problem a user may have and a solution. You could instead have a 3-4 line explanation printed out with the error, but if it is too long for that, use the docs and leave a note in the error message saying something like <b><samp>Check out our website and docs for more</samp></b><br/><br></li>

<li>Also, please comment unclear code for maintainability. Don't overcomment, but commenting too much is better than bad, confusing code.

<br>

<blockquote>Comment on code that may be misunderstood, but don't comment things like <samp><br># This is an if statement</samp>
or
<samp><br># This  is list comprehension</samp>
</blockquote>
</li>

<br>

<li>Also, for maintaiability's sake, please give variables meaningful names. Follow the following structure for variable naming.

<br>

- **`CapitalizedCamelCase`** - class names
- **`lower_case_with_underscores`** - functions and methods
- **`normalCamelCase`**  or **`lower_case_with_underscores`** - variables
- **`CAPITALIZED`** - constants

</li>

<li>Please use **black** for formatting. This is set as default in the vscode workspace config.</li>
</ol>
