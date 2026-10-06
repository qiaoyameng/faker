from typing import Optional, Tuple, Union

from ...utils.unique import FiniteDomain, PrefixedNumberDomain
from .. import BaseProvider

localized = True

PrefixType = Tuple[Union[int, str, Tuple[Union[int, str], ...]], ...]


def _ean_check_digit(body: str) -> int:
    """Compute the EAN check digit for a string of body digits."""
    weighted_sum = sum(int(digit) * (3 if index % 2 == 0 else 1) for index, digit in enumerate(reversed(body)))
    return (10 - weighted_sum % 10) % 10


def _normalize_ean_prefix(prefix: Union[int, str, Tuple[Union[int, str], ...]]) -> Optional[str]:
    """Normalize a single EAN prefix to its digit string, or ``None`` if not possible."""
    if isinstance(prefix, str):
        parts = list(prefix)
    elif isinstance(prefix, (tuple, list)):
        parts = [str(part) for part in prefix]
    else:
        parts = [str(prefix)]
    if not all(part.isdigit() and len(part) == 1 for part in parts):
        return None
    return "".join(parts)


class EanDomain(FiniteDomain[str]):
    """The finite domain of EAN codes of a fixed length sharing one fixed prefix."""

    def __init__(self, length: int, prefix: str = "") -> None:
        if length not in (8, 13):
            raise ValueError("length can only be 8 or 13")
        if len(prefix) >= length:
            raise ValueError("prefix must be shorter than the code")
        self.length = length
        self.prefix = prefix
        self._body = PrefixedNumberDomain(prefix, length - 1 - len(prefix))

    @property
    def size(self) -> int:
        return self._body.size

    def unrank(self, index: int) -> str:
        body = self._body.unrank(index)
        return body + str(_ean_check_digit(body))

    def rank(self, value: str) -> int:
        text = str(value)
        if len(text) != self.length or not text.isdigit():
            raise ValueError(f"{value!r} is not in the domain")
        body, check_digit = text[:-1], int(text[-1])
        if _ean_check_digit(body) != check_digit:
            raise ValueError(f"{value!r} is not in the domain")
        return self._body.rank(body)


class Provider(BaseProvider):
    """Implement default barcode provider for Faker.

    Sources:

    - https://gs1.org/standards/id-keys/company-prefix
    """

    local_prefixes: PrefixType = ()

    def _ean(self, length: int = 13, prefixes: PrefixType = ()) -> str:
        if length not in (8, 13):
            raise AssertionError("length can only be 8 or 13")

        code = [self.random_digit() for _ in range(length - 1)]

        if prefixes:
            prefix: str = self.random_element(prefixes)  # type: ignore[assignment]
            code[: len(prefix)] = map(int, prefix)

        code.append(_ean_check_digit("".join(str(digit) for digit in code)))

        return "".join(str(x) for x in code)

    def _ean_unique_domain(self, length: int = 13, prefixes: PrefixType = ()) -> Optional[FiniteDomain[str]]:
        return self._build_ean_domain(length, prefixes)

    def _ean8_unique_domain(self, prefixes: PrefixType = ()) -> Optional[FiniteDomain[str]]:
        return self._build_ean_domain(8, prefixes)

    def _ean13_unique_domain(self, prefixes: PrefixType = ()) -> Optional[FiniteDomain[str]]:
        return self._build_ean_domain(13, prefixes)

    def _localized_ean_unique_domain(self, length: int = 13) -> Optional[FiniteDomain[str]]:
        return self._build_ean_domain(length, self.local_prefixes)

    def _localized_ean8_unique_domain(self) -> Optional[FiniteDomain[str]]:
        return self._build_ean_domain(8, self.local_prefixes)

    def _localized_ean13_unique_domain(self) -> Optional[FiniteDomain[str]]:
        return self._build_ean_domain(13, self.local_prefixes)

    def _build_ean_domain(self, length: int, prefixes: PrefixType) -> Optional[FiniteDomain[str]]:
        """Declare the finite value domain of an EAN variant for the ``unique`` proxy.

        A domain can only be declared when the prefix is fixed, i.e. when no
        prefixes or exactly one prefix is given.  Otherwise ``None`` is
        returned and the ``unique`` proxy falls back to its regular
        retry-based behavior.
        """
        if length not in (8, 13):
            return None
        if not prefixes:
            return EanDomain(length)
        if len(prefixes) != 1:
            return None
        prefix = _normalize_ean_prefix(prefixes[0])
        if prefix is None or len(prefix) >= length:
            return None
        return EanDomain(length, prefix)

    def ean(self, length: int = 13, prefixes: PrefixType = ()) -> str:
        """Generate an EAN barcode of the specified ``length``.

        The value of ``length`` can only be ``8`` or ``13`` (default) which will
        create an EAN-8 or an EAN-13 barcode respectively.

        If a value for ``prefixes`` is specified, the result will begin with one
        of the sequences in ``prefixes``.

        :sample: length=13
        :sample: length=8
        :sample: prefixes=('00',)
        :sample: prefixes=('45', '49')
        """
        return self._ean(length, prefixes=prefixes)

    def ean8(self, prefixes: PrefixType = ()) -> str:
        """Generate an EAN-8 barcode.

        This method uses |ean| under the hood with the ``length`` argument
        explicitly set to ``8``.

        If a value for ``prefixes`` is specified, the result will begin with one
        of the sequences in ``prefixes``.

        :sample:
        :sample: prefixes=('00',)
        :sample: prefixes=('45', '49')
        """
        return self._ean(8, prefixes=prefixes)

    def ean13(self, prefixes: PrefixType = ()) -> str:
        """Generate an EAN-13 barcode.

        This method uses |ean| under the hood with the ``length`` argument
        explicitly set to ``13``.

        If a value for ``prefixes`` is specified, the result will begin with one
        of the sequences in ``prefixes``.

        .. note::
           Codes starting with a leading zero are treated specially in some
           barcode readers. For more information on compatibility with UPC-A
           codes, see |EnUsBarcodeProvider.ean13|.

        :sample:
        :sample: prefixes=('00',)
        :sample: prefixes=('45', '49')
        """
        return self._ean(13, prefixes=prefixes)

    def localized_ean(self, length: int = 13) -> str:
        """Generate a localized EAN barcode of the specified ``length``.

        The value of ``length`` can only be ``8`` or ``13`` (default) which will
        create an EAN-8 or an EAN-13 barcode respectively.

        This method uses the standard barcode provider's |ean| under the hood
        with the ``prefixes`` argument explicitly set to ``local_prefixes`` of
        a localized barcode provider implementation.

        :sample:
        :sample: length=13
        :sample: length=8
        """
        return self._ean(length, prefixes=self.local_prefixes)

    def localized_ean8(self) -> str:
        """Generate a localized EAN-8 barcode.

        This method uses |localized_ean| under the hood with the ``length``
        argument explicitly set to ``8``.
        """
        return self.localized_ean(8)

    def localized_ean13(self) -> str:
        """Generate a localized EAN-13 barcode.

        This method uses |localized_ean| under the hood with the ``length``
        argument explicitly set to ``13``.
        """
        return self.localized_ean(13)
