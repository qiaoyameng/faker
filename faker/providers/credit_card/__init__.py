from collections import OrderedDict
from collections.abc import Iterable as IterableABC
from typing import Collection, Dict, List, Optional, Sequence, Union

from ...typing import CardType, CreditCard, DateParseType
from .. import BaseProvider

localized = True

DigitMask = Sequence[Union[str, int, Collection[Union[str, int]]]]
LUHN_DIGIT_VALUES = (
    tuple(range(10)),
    (0, 2, 4, 6, 8, 1, 3, 5, 7, 9),
)


def _normalize_digit_mask(mask: Union[str, DigitMask]) -> List[tuple[int, ...]]:
    if isinstance(mask, bytes | bytearray):
        raise TypeError("The credit card mask must be a string or sequence.")

    if isinstance(mask, str):
        positions: IterableABC = mask
    elif isinstance(mask, Sequence):
        positions = mask
    else:
        raise TypeError("The credit card mask must be a string or sequence.")

    digit_sets: List[tuple[int, ...]] = []
    for position, allowed in enumerate(positions):
        digits: set[int] = set()

        def add_digit(value: Union[str, int]) -> None:
            if isinstance(value, bool):
                raise ValueError(f"Position {position} contains an invalid digit.")
            if isinstance(value, int):
                digit = value
            elif isinstance(value, str) and len(value) == 1 and value in "0123456789":
                digit = int(value)
            else:
                raise ValueError(f"Position {position} contains an invalid digit.")

            if not 0 <= digit <= 9:
                raise ValueError(f"Position {position} contains an invalid digit.")
            digits.add(digit)

        if isinstance(allowed, bool):
            raise ValueError(f"Position {position} contains an invalid digit.")
        if isinstance(allowed, int):
            add_digit(allowed)
        elif isinstance(allowed, str):
            for char in allowed:
                if char == "#":
                    digits.update(range(10))
                else:
                    add_digit(char)
        elif isinstance(allowed, IterableABC) and not isinstance(allowed, dict | bytes | bytearray):
            for value in allowed:
                add_digit(value)
        else:
            raise ValueError(f"Position {position} contains an invalid digit set.")

        if not digits:
            raise ValueError(f"Position {position} must allow at least one digit.")
        digit_sets.append(tuple(sorted(digits)))

    if not digit_sets:
        raise ValueError("The credit card mask must contain at least one position.")
    return digit_sets


def _luhn_suffix_counts(digit_sets: Sequence[tuple[int, ...]]) -> List[List[int]]:
    suffix_counts: List[List[int]] = [[0] * 10 for _ in range(len(digit_sets) + 1)]
    suffix_counts[-1][0] = 1

    for position in range(len(digit_sets) - 1, -1, -1):
        weight_index = 0 if (len(digit_sets) - 1 - position) % 2 == 0 else 1
        next_counts = suffix_counts[position + 1]
        current_counts = suffix_counts[position]

        for needed in range(10):
            current_counts[needed] = sum(
                next_counts[(needed - LUHN_DIGIT_VALUES[weight_index][digit]) % 10]
                for digit in digit_sets[position]
            )

    return suffix_counts


class Provider(BaseProvider):
    """Implement default credit card provider for Faker.

    For all methods that take ``card_type`` as an argument, a random card type
    will be used if the supplied value is ``None``. The list of valid card types
    includes ``'amex'``, ``'diners'``, ``'discover'``, ``'jcb'``, ``'jcb15'``,
    ``'jcb16'``, ``'maestro'``, ``'mastercard'``, ``'visa'``, ``'visa13'``,
    ``'visa16'``, and ``'visa19'``.

    Sources:

    - https://en.wikipedia.org/wiki/Payment_card_number#Issuer_identification_number_.28IIN.29
    - https://www.regular-expressions.info/creditcard.html
    - https://creditcardjs.com/credit-card-type-detection
    """

    prefix_maestro: List[str] = [
        "5018",
        "5020",
        "5038",
        "56##",
        "57##",
        "58##",
        "6304",
        "6759",
        "6761",
        "6762",
        "6763",
        "0604",
        "6390",
    ]
    prefix_mastercard: List[str] = [
        "51",
        "52",
        "53",
        "54",
        "55",
        "222%",
        "223",
        "224",
        "225",
        "226",
        "227",
        "228",
        "229",
        "23",
        "24",
        "25",
        "26",
        "270",
        "271",
        "2720",
    ]
    prefix_visa: List[str] = ["4"]
    prefix_amex: List[str] = ["34", "37"]
    prefix_discover: List[str] = ["6011", "65"]
    prefix_diners: List[str] = ["300", "301", "302", "303", "304", "305", "36", "38"]
    prefix_jcb16: List[str] = ["35"]
    prefix_jcb15: List[str] = ["2131", "1800"]

    credit_card_types: Dict[str, CreditCard] = OrderedDict(
        (
            ("maestro", CreditCard("Maestro", prefix_maestro, 12, security_code="CVV")),
            (
                "mastercard",
                CreditCard("Mastercard", prefix_mastercard, 16, security_code="CVV"),
            ),
            ("visa16", CreditCard("VISA 16 digit", prefix_visa)),
            ("visa13", CreditCard("VISA 13 digit", prefix_visa, 13)),
            ("visa19", CreditCard("VISA 19 digit", prefix_visa, 19)),
            (
                "amex",
                CreditCard(
                    "American Express",
                    prefix_amex,
                    15,
                    security_code="CID",
                    security_code_length=4,
                ),
            ),
            ("discover", CreditCard("Discover", prefix_discover)),
            ("diners", CreditCard("Diners Club / Carte Blanche", prefix_diners, 14)),
            ("jcb15", CreditCard("JCB 15 digit", prefix_jcb15, 15)),
            ("jcb16", CreditCard("JCB 16 digit", prefix_jcb16)),
        )
    )
    credit_card_types["visa"] = credit_card_types["visa16"]
    credit_card_types["jcb"] = credit_card_types["jcb16"]

    luhn_lookup = {
        "0": 0,
        "1": 2,
        "2": 4,
        "3": 6,
        "4": 8,
        "5": 1,
        "6": 3,
        "7": 5,
        "8": 7,
        "9": 9,
    }

    def credit_card_provider(self, card_type: Optional[CardType] = None) -> str:
        """Generate a credit card provider name."""
        if card_type is None:
            card_type = self.random_element(self.credit_card_types.keys())  # type: ignore[assignment]
        return self._credit_card_type(card_type).name

    def credit_card_number(self, card_type: Optional[CardType] = None) -> str:
        """Generate a valid credit card number."""
        card = self._credit_card_type(card_type)
        prefix: str = self.random_element(card.prefixes)
        number = self._generate_number(self.numerify(prefix), card.length)
        return number

    def credit_card_number_from_mask(self, mask: Union[str, DigitMask] = "#" * 16) -> str:
        """Generate a valid credit card number matching the per-digit mask.

        A string position accepts those exact digits, ``#`` allows any digit,
        and any iterable position may contain integers or one-character digit
        strings. Each complete valid number has equal probability.
        """
        digit_sets = _normalize_digit_mask(mask)
        suffix_counts = _luhn_suffix_counts(digit_sets)
        total = suffix_counts[0][0]

        if total == 0:
            raise ValueError("No credit card numbers satisfy the supplied mask.")

        number = []
        needed = 0
        remaining = total

        for position, allowed_digits in enumerate(digit_sets):
            weight_index = 0 if (len(digit_sets) - 1 - position) % 2 == 0 else 1
            digit_values = LUHN_DIGIT_VALUES[weight_index]
            weights = [
                suffix_counts[position + 1][(needed - digit_values[digit]) % 10]
                for digit in allowed_digits
            ]

            choice = self.generator.random.randrange(remaining)
            cumulative = 0
            selected = allowed_digits[0]
            next_needed = 0

            for digit, branch_weight in zip(allowed_digits, weights):
                cumulative += branch_weight
                if choice < cumulative:
                    selected = digit
                    next_needed = (needed - digit_values[digit]) % 10
                    break

            number.append(str(selected))
            needed = next_needed
            remaining = suffix_counts[position + 1][needed]

        return "".join(number)

    def credit_card_number_with_digit_sets(self, digit_sets: DigitMask = ("#",) * 16) -> str:
        """Generate a valid number from a sequence of per-digit digit sets."""
        return self.credit_card_number_from_mask(digit_sets)

    def credit_card_number_from_digit_sets(self, digit_sets: DigitMask = ("#",) * 16) -> str:
        """Generate a valid number from a sequence of per-digit digit sets."""
        return self.credit_card_number_from_mask(digit_sets)

    def credit_card_number_count(self, mask: Union[str, DigitMask] = "#" * 16) -> int:
        """Return the number of Luhn-valid numbers allowed by the mask."""
        return _luhn_suffix_counts(_normalize_digit_mask(mask))[0][0]

    def credit_card_number_mask_count(self, mask: Union[str, DigitMask] = "#" * 16) -> int:
        """Return the number of Luhn-valid numbers allowed by the mask."""
        return self.credit_card_number_count(mask)

    def credit_card_expire(
        self,
        start: DateParseType = "now",
        end: DateParseType = "+10y",
        date_format: str = "%m/%y",
    ) -> str:
        """Generate a credit card expiry date.

        This method uses |date_time_between| under the hood to generate the
        expiry date, so the ``start`` and ``end`` arguments work in the same way
        here as it would in that method. For the actual formatting of the expiry
        date, |strftime| is used and ``date_format`` is simply passed
        to that method.
        """
        expire_date = self.generator.date_time_between(start, end)
        return expire_date.strftime(date_format)

    def credit_card_full(self, card_type: Optional[CardType] = None) -> str:
        """Generate a set of credit card details."""
        card = self._credit_card_type(card_type)

        tpl = "{provider}\n{owner}\n{number} {expire_date}\n{security}: {security_nb}\n"

        tpl = tpl.format(
            provider=card.name,
            owner=self.generator.parse("{{first_name}} {{last_name}}"),
            number=self.credit_card_number(card),
            expire_date=self.credit_card_expire(),
            security=card.security_code,
            security_nb=self.credit_card_security_code(card),
        )

        return self.generator.parse(tpl)

    def credit_card_security_code(self, card_type: Optional[CardType] = None) -> str:
        """Generate a credit card security code."""
        sec_len = self._credit_card_type(card_type).security_code_length
        return self.numerify("#" * sec_len)

    def _credit_card_type(self, card_type: Optional[CardType] = None) -> CreditCard:
        """Generate a random CreditCard instance of the specified card type."""
        if card_type is None:
            card_type = self.random_element(self.credit_card_types.keys())  # type: ignore[assignment]
        elif isinstance(card_type, CreditCard):
            return card_type
        return self.credit_card_types[card_type]  # type: ignore[index]

    def _generate_number(self, prefix: str, length: int) -> str:
        """Generate a credit card number.

        The ``prefix`` argument is the start of the CC number as a string which
         may contain any number of digits. The ``length`` argument is the length
         of the CC number to generate which is typically 13 or 16.
        """
        number = prefix
        # Generate random char digits
        number += "#" * (length - len(prefix) - 1)
        number = self.numerify(number)
        reverse = number[::-1]
        # Calculate sum
        tot = 0
        pos = 0
        while pos < length - 1:
            tot += Provider.luhn_lookup[reverse[pos]]
            if pos != (length - 2):
                tot += int(reverse[pos + 1])
            pos += 2
        # Calculate check digit
        check_digit = (10 - (tot % 10)) % 10
        number += str(check_digit)
        return number
