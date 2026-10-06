from collections.abc import Mapping
from typing import Any


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


class InfeasibleQuotaError(BaseFakerException, ValueError):
    """The supplied categorical quotas cannot be satisfied simultaneously."""

    def __init__(
        self,
        row_subset: tuple[Any, ...],
        column_subset: tuple[Any, ...],
        row_quotas: Mapping[Any, int],
        column_quotas: Mapping[Any, int],
        shortfall: int,
    ) -> None:
        self.row_subset: tuple[Any, ...] = row_subset
        self.column_subset: tuple[Any, ...] = column_subset
        self.row_quotas: Mapping[Any, int] = row_quotas
        self.column_quotas: Mapping[Any, int] = column_quotas
        self.shortfall = shortfall

        supply = sum(row_quotas[row] for row in row_subset)
        demand = sum(column_quotas[column] for column in column_subset)
        super().__init__(
            "Quota constraints are infeasible: the row supply for "
            f"{list(row_subset)!r} is {supply} and the usable column demand "
            f"for {list(column_subset)!r} is {demand}; "
            f"the verifiable gap is {shortfall}."
        )
