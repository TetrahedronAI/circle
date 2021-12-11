# Python Coding Style

## Testing
We use the built in **unittest** module for testing. Please use it too, since it appears as part of the test explorer in the sidebar.

Also, lease use the provided testing utils for testing, as seen below...

The decorator that logs the files is as below. **(This requires you to already have a folder called *`logs`*)**.

```python
from easyneuron._testutils import log_errors
```

Then, for every **unittest** test, simply decorate it with `@log_errors`.