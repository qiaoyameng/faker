"""Finite value domains used by the ``unique`` proxy.

A :class:`ValueDomain` describes a *finite* set of possible outputs of a
provider method without materializing them.  The set is expressed through its
:attr:`ValueDomain.size` (the number of distinct values) together with a pair
of complementary operations:

- :meth:`ValueDomain.rank` maps a generated value to its index in ``[0, size)``;
- :meth:`ValueDomain.unrank` maps an index in ``[0, size)`` back to a value.

This lets the unique machinery perform uniform sampling *without replacement*
as a permutation of the indices, so it never needs to retry on collisions and
its memory use grows with the number of values actually drawn rather than with
the size of the domain.

Providers opt in by decorating a method with :func:`finite_domain` and giving
it a domain factory.  Provider methods that do not declare a finite domain keep
the original retry based uniqueness behavior, which also covers infinite or
unknown value ranges.
"""

from __future__ import annotations

import abc
import functools

from typing import Any, Callable, Type

# Attribute set on provider methods decorated with :func:`finite_domain`.
FINITE_DOMAIN_ATTR = "__faker_finite_domain__"


class ValueDomain(abc.ABC):
    """A finite, rankable set of provider output values."""

    @property
    @abc.abstractmethod
    def size(self) -> int:
        """Number of distinct values in the domain."""

    @abc.abstractmethod
    def unrank(self, index: int) -> Any:
        """Return the value at ``index`` in ``[0, self.size)``."""

    @abc.abstractmethod
    def rank(self, value: Any) -> int:
        """Return the index of ``value`` within ``[0, self.size)``."""

    @property
    def exhausted(self) -> bool:
        return self.size <= 0


class IntegerRangeDomain(ValueDomain):
    """Domain of integers, equivalent to ``range(min, max + 1, step)``."""

    def __init__(self, min: int = 0, max: int = 9999, step: int = 1) -> None:
        if step == 0:
            raise ValueError("step cannot be zero")
        self.min = int(min)
        self.max = int(max)
        self.step = int(step)
        # len(range(min, max + 1, step)): the range is inclusive at both ends
        stop = self.max + 1 if self.step > 0 else self.max - 1
        adjustment = self.step - 1 if self.step > 0 else self.step + 1
        size = (stop - self.min + adjustment) // self.step
        self._size = size if size > 0 else 0

    @property
    def size(self) -> int:
        return self._size

    def unrank(self, index: int) -> int:
        if not 0 <= index < self._size:
            raise IndexError("index out of range for this domain")
        return self.min + index * self.step

    def rank(self, value: Any) -> int:
        value = int(value)
        if (value - self.min) % self.step != 0:
            raise ValueError(f"{value!r} is not part of this domain")
        index = (value - self.min) // self.step
        if not 0 <= index < self._size:
            raise ValueError(f"{value!r} is not part of this domain")
        return index


class NumericCodeDomain(ValueDomain):
    """Domain of fixed length numeric codes sharing a constant prefix.

    Every value is the concatenation of ``prefix`` with exactly ``digits``
    decimal digits, zero padded when necessary (e.g. ``"SKU-042"``).  Only the
    numeric portion varies, so the domain contains ``10 ** digits`` values.
    """

    _radix = 10

    def __init__(self, prefix: str = "", digits: int = 4) -> None:
        if not isinstance(prefix, str):
            raise TypeError("prefix must be a string")
        if not isinstance(digits, int) or isinstance(digits, bool) or digits < 0:
            raise ValueError("digits must be a non-negative integer")
        self.prefix = prefix
        self.digits = digits
        self._size = self._radix**digits

    @property
    def size(self) -> int:
        return self._size

    def unrank(self, index: int) -> str:
        if not 0 <= index < self._size:
            raise IndexError("index out of range for this domain")
        if self.digits == 0:
            return self.prefix
        return self.prefix + str(index).zfill(self.digits)

    def rank(self, value: Any) -> int:
        if not isinstance(value, str) or not value.startswith(self.prefix):
            raise ValueError(f"{value!r} is not part of this domain")
        number_part = value[len(self.prefix) :]
        if self.digits == 0:
            if number_part != "":
                raise ValueError(f"{value!r} is not part of this domain")
            return 0
        if len(number_part) != self.digits or not number_part.isascii() or not number_part.isdigit():
            raise ValueError(f"{value!r} is not part of this domain")
        return int(number_part)


def finite_domain(domain_cls: Type[ValueDomain]) -> Callable[[Callable], Callable]:
    """Decorator declaring the finite value domain of a provider method.

    ``domain_cls`` is constructed with the same positional and keyword
    arguments passed to the provider method.  The decorated method is returned
    unchanged; the decorator only attaches the domain factory so that
    ``fake.unique`` can sample without replacement.
    """

    def decorator(function: Callable) -> Callable:
        @functools.wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return function(*args, **kwargs)

        setattr(wrapper, FINITE_DOMAIN_ATTR, domain_cls)
        return wrapper

    return decorator
