import warnings


class deprecated(object):
    """Mark a function or class as deprecated (decorator).

    Issue a warning when the function is called/the class is instantiated and
    adds a warning to the docstring.

    The optional extra argument will be appended to the deprecation message
    and the docstring. Note: to use this with the default value for extra, put
    in an empty of parentheses::

        >>> from pyriemann.utils import deprecated
        >>> deprecated()
        <pyriemann.utils.docs.deprecated object at ...>
        >>> @deprecated()
        ... def some_function(): pass


    Parameters
    ----------
    extra: string
        To be added to the deprecation messages.
    """

    # Borrowed from MNE:
    # https://mne.tools/stable/generated/mne.utils.deprecated.html

    def __init__(self, extra=""):
        self.extra = extra

    def __call__(self, obj):
        """Call.
        Parameters
        ----------
        obj : object
            Object to call.
        """
        if isinstance(obj, type):
            return self._decorate_class(obj)
        else:
            return self._decorate_fun(obj)

    def _decorate_class(self, cls):
        pass

    def _decorate_fun(self, fun):
        """Decorate function fun."""
        pass

    def _update_doc(self, olddoc):
        pass
