"""Finite value domains for the ``unique`` proxy.

Some provider methods draw values from a finite value domain that can be
described by its size together with ``rank``/``unrank`` mappings, without
materializing every possible value.  The ``unique`` proxy uses such
declarations to sample uniformly and without replacement in ``O(k)`` memory
(``k`` being the number of values drawn so far), instead of retrying until a
fresh value happens to come up.
"""

from abc import ABC, abstractmethod
from random import Random
from typing import Dict, Generic, TypeVar

T = TypeVar("T")


class FiniteDomain(ABC, Generic[T]):
    """A finite set of values addressable by rank.

    Implementations must provide the number of values in the domain along
    with a bijection between the values and the integers
    ``range(0, domain.size)``.  Values are never required to be enumerated
    up front, so domains of any size can be declared cheaply.
    """

    @property
    @abstractmethod
    def size(self) -> int:
        """The number of distinct values in the domain."""
        raise NotImplementedError

    @abstractmethod
    def unrank(self, index: int) -> T:
        """Return the value stored at ``index``, where ``0 <= index < size``."""
        raise NotImplementedError

    @abstractmethod
    def rank(self, value: T) -> int:
        """Return the index of ``value``, raising ``ValueError`` if it is not in the domain."""
        raise NotImplementedError


class IntegerRangeDomain(FiniteDomain[int]):
    """The integers of ``range(minimum, maximum + 1, step)``."""

    def __init__(self, minimum: int, maximum: int, step: int = 1) -> None:
        if step < 1:
            raise ValueError("step must be a positive integer")
        self.minimum = minimum
        self.maximum = maximum
        self.step = step

    @property
    def size(self) -> int:
        if self.maximum < self.minimum:
            return 0
        return (self.maximum - self.minimum) // self.step + 1

    def unrank(self, index: int) -> int:
        if not 0 <= index < self.size:
            raise IndexError(f"index {index} is out of range for a domain of size {self.size}")
        return self.minimum + index * self.step

    def rank(self, value: int) -> int:
        offset = value - self.minimum
        if offset < 0 or offset % self.step != 0:
            raise ValueError(f"{value} is not in the domain")
        index = offset // self.step
        if index >= self.size:
            raise ValueError(f"{value} is not in the domain")
        return index

    def __repr__(self) -> str:
        return f"{type(self).__name__}(minimum={self.minimum}, maximum={self.maximum}, step={self.step})"


class PrefixedNumberDomain(FiniteDomain[str]):
    """Fixed literal prefix followed by a fixed-width zero-padded decimal number."""

    def __init__(self, prefix: str, width: int) -> None:
        if width < 0:
            raise ValueError("width must be a non-negative integer")
        self.prefix = prefix
        self.width = width

    @property
    def size(self) -> int:
        return 10**self.width

    def unrank(self, index: int) -> str:
        if not 0 <= index < self.size:
            raise IndexError(f"index {index} is out of range for a domain of size {self.size}")
        body = f"{index:0{self.width}d}" if self.width else ""
        return f"{self.prefix}{body}"

    def rank(self, value: str) -> int:
        text = str(value)
        if not text.startswith(self.prefix):
            raise ValueError(f"{value!r} is not in the domain")
        body = text[len(self.prefix) :]
        if len(body) != self.width or not body.isdigit():
            raise ValueError(f"{value!r} is not in the domain")
        return int(body)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(prefix={self.prefix!r}, width={self.width})"


class DomainSampler(Generic[T]):
    """Uniform sampling without replacement from a :class:`FiniteDomain`.

    A lazy Fisher-Yates shuffle is used, so memory grows with the number of
    values drawn, never with the size of the domain.
    """

    def __init__(self, domain: FiniteDomain[T], random: Random) -> None:
        self.domain = domain
        self.random = random
        self._swaps: Dict[int, int] = {}
        self._drawn = 0

    @property
    def drawn(self) -> int:
        """The number of values drawn so far."""
        return self._drawn

    @property
    def exhausted(self) -> bool:
        """Whether every value of the domain has been drawn."""
        return self._drawn >= self.domain.size

    def sample(self) -> T:
        """Draw the next value uniformly from the values not yet drawn."""
        if self.exhausted:
            raise IndexError("the domain is exhausted")
        position = self._drawn
        index = self.random.randrange(position, self.domain.size)
        # Swap the implicit permutation entries at ``position`` and ``index``,
        # storing only entries that differ from the identity permutation.
        value_index = self._swaps.pop(index, index)
        displaced = self._swaps.pop(position, position)
        if value_index != position:
            self._swaps[position] = value_index
        if displaced != index:
            self._swaps[index] = displaced
        self._drawn = position + 1
        return self.domain.unrank(value_index)
