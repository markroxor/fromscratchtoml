# define Python user-defined exceptions


class Error(Exception):
    pass


class ModelNotFittedError(Error):
    pass


class ValueTooLargeError(Error):
    pass
