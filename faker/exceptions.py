class BaseFakerException(Exception):
    """The base exception for all Faker exceptions."""


class UniquenessException(BaseFakerException):
    """To avoid infinite loops, after a certain number of attempts,
    the "unique" attribute of the Proxy will throw this exception.
    """


class UnsupportedFeature(BaseFakerException):
    """The requested feature is not available on this system."""

    def __init__(self, msg: str, name: str) -> None:
        self.name = name
        super().__init__(msg)


class TextGenerationException(BaseFakerException):
    """Base exception for text generation failures."""


class TextGenerationBudgetExceeded(TextGenerationException):
    """The configured search budget was exhausted before generation finished."""


class ImpossibleTextLength(TextGenerationException):
    """No combination of the supplied fragments can have the requested length."""
