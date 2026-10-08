"""Exceptions that signal a problem with the user's data, as opposed to an internal bug."""


class InputError(ValueError):
    """The uploaded or supplied data is unusable (missing label, mismatched columns, etc.).

    The API maps this to HTTP 400 and the CLI to exit code 2. Any other exception is an
    internal error.
    """
