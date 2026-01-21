"""Custom exceptions for kmcomp."""


class PathIsNoneError(Exception):
    """Raised when an expected directory is None."""

    def __init__(self, path, message="Path is None."):
        super().__init__(message)
        self.path = path

    def __str__(self):
        return f"{self.path} -> {self.args[0]}"


class NoSupportedImageFilesError(Exception):
    """Raised when an expected directory is empty."""

    def __init__(
        self, path, message="Directory does not contain a supported image file."
    ):
        super().__init__(message)
        self.path = path

    def __str__(self):
        return f"{self.path} -> {self.args[0]}"


class UnsupportedImageTypeError(Exception):
    """Raised when the image file is not supported."""

    def __init__(self, path, message="Image file is not supported."):
        super().__init__(message)
        self.path = path

    def __str__(self):
        return f"{self.path} -> {self.args[0]}"


class UnsupportedFileTypeError(Exception):
    """Raised when the file is not supported."""

    def __init__(self, path, message="File type is not supported."):
        super().__init__(message)
        self.path = path

    def __str__(self):
        return f"{self.path} -> {self.args[0]}"


class NotDirectoryError(Exception):
    """Raised when an expected path is not a directory."""

    def __init__(self, path, message="Target path is not a directory."):
        super().__init__(message)
        self.path = path

    def __str__(self):
        return f"{self.path} -> {self.args[0]}"


class NotFileError(Exception):
    """Raised when an expected path is not a file."""

    def __init__(self, path, message="Target path is not a file."):
        super().__init__(message)
        self.path = path

    def __str__(self):
        return f"{self.path} -> {self.args[0]}"


class PathDoesNotExistError(Exception):
    """Raised when an expected path does not exist."""

    def __init__(self, path, message="Target path does not exist."):
        super().__init__(message)
        self.path = path

    def __str__(self):
        return f"{self.path} -> {self.args[0]}"


class Exceptions:
    """Custom exceptions"""

    PathIsNoneError = PathIsNoneError
    PathDoesNotExistError = PathDoesNotExistError
    NoSupportedImageFilesError = NoSupportedImageFilesError
    UnsupportedImageTypeError = UnsupportedImageTypeError
    UnsupportedFileTypeError = UnsupportedFileTypeError
    NotDirectoryError = NotDirectoryError
    NotFileError = NotFileError
