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


class InfeasibleQuotaException(BaseFakerException):
    """The two quota sets cannot be satisfied under the forbidden combinations."""

    def __init__(self, msg: str, certificate: dict | None = None) -> None:
        self.certificate = certificate or {}
        super().__init__(msg)
