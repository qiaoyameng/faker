import itertools
import re

from typing import Pattern

import pytest

from faker.providers.bank.ru_RU import Provider as RuRuBankProvider
from faker.providers.bank.uk_UA import Provider as UkUaBankProvider
from faker.providers.credit_card import Provider as CreditCardProvider


def luhn_valid(number: str) -> bool:
    total = 0
    for pos, char in enumerate(reversed(number)):
        if pos % 2 == 1:
            total += CreditCardProvider.luhn_lookup[char]
        else:
            total += int(char)
    return total % 10 == 0


def exhaustive_valid_numbers(mask) -> set:
    positions = ["0123456789" if position is None else str(position) for position in mask]
    return {
        "".join(combination)
        for combination in itertools.product(*positions)
        if luhn_valid("".join(combination))
    }


class TestCreditCardProvider:
    """Test credit card provider methods"""

    mastercard_pattern: Pattern = re.compile(
        r"(?:5[1-5][0-9]{2}|222[1-9]|22[3-9][0-9]|2[3-6][0-9]{2}|27[01][0-9]|2720)[0-9]{12}",
    )
    visa_pattern: Pattern = re.compile(r"4[0-9]{12}([0-9]{3}){0,2}")
    discover_pattern: Pattern = re.compile(r"6(?:011|5[0-9]{2})[0-9]{12}")
    diners_club_pattern: Pattern = re.compile(r"3(?:0[0-5]|[68][0-9])[0-9]{11}")
    jcb_pattern: Pattern = re.compile(r"(?:2131|1800|35\d{3})\d{11}")

    def test_mastercard(self, faker, num_samples):
        provider = CreditCardProvider(faker)
        for prefix in provider.prefix_mastercard:
            for _ in range(num_samples):
                number = provider._generate_number(prefix, 16)
                assert len(number) == 16
                assert self.mastercard_pattern.fullmatch(number)

    def test_visa13(self, faker, num_samples):
        provider = CreditCardProvider(faker)
        for prefix in provider.prefix_visa:
            for _ in range(num_samples):
                number = provider._generate_number(prefix, 13)
                assert len(number) == 13
                assert self.visa_pattern.fullmatch(number)

    def test_visa16(self, faker, num_samples):
        provider = CreditCardProvider(faker)
        for prefix in provider.prefix_visa:
            for _ in range(num_samples):
                number = provider._generate_number(prefix, 16)
                assert len(number) == 16
                assert self.visa_pattern.fullmatch(number)

    def test_visa19(self, faker, num_samples):
        provider = CreditCardProvider(faker)
        for prefix in provider.prefix_visa:
            for _ in range(num_samples):
                number = provider._generate_number(prefix, 19)
                assert len(number) == 19
                assert self.visa_pattern.fullmatch(number)

    def test_discover(self, faker, num_samples):
        provider = CreditCardProvider(faker)
        for prefix in provider.prefix_discover:
            for _ in range(num_samples):
                number = provider._generate_number(prefix, 16)
                assert len(number) == 16
                assert self.discover_pattern.fullmatch(number)

    def test_diners_club(self, faker, num_samples):
        provider = CreditCardProvider(faker)
        for prefix in provider.prefix_diners:
            for _ in range(num_samples):
                number = provider._generate_number(prefix, 14)
                assert len(number) == 14
                assert self.diners_club_pattern.fullmatch(number)

    def test_jcb16(self, faker, num_samples):
        provider = CreditCardProvider(faker)
        for prefix in provider.prefix_jcb16:
            for _ in range(num_samples):
                number = provider._generate_number(prefix, 16)
                assert len(number) == 16
                assert self.jcb_pattern.fullmatch(number)

    def test_jcb15(self, faker, num_samples):
        provider = CreditCardProvider(faker)
        for prefix in provider.prefix_jcb15:
            for _ in range(num_samples):
                number = provider._generate_number(prefix, 15)
                assert len(number) == 15
                assert self.jcb_pattern.fullmatch(number)


class TestPtPt:
    """Test pt_PT credit card provider methods"""

    visa_pattern: Pattern = re.compile(r"4[0-9]{15}")
    mastercard_pattern: Pattern = re.compile(r"5[1-5][0-9]{14}")
    maestro_pattern: Pattern = re.compile(r"(50|67)[0-9]{14}")

    def test_visa(self, faker, num_samples):
        for _ in range(num_samples):
            number = faker.credit_card_number("visa")
            assert self.visa_pattern.fullmatch(number)

    def test_mastercard(self, faker, num_samples):
        for _ in range(num_samples):
            number = faker.credit_card_number("mastercard")
            assert self.mastercard_pattern.fullmatch(number)

    def test_maestro(self, faker, num_samples):
        for _ in range(num_samples):
            number = faker.credit_card_number("maestro")
            assert self.maestro_pattern.fullmatch(number)


class TestRuRu:
    """Test ru_RU credit card provider methods"""

    visa_pattern: Pattern = re.compile(r"4[0-9]{15}")
    mastercard_pattern: Pattern = re.compile(
        r"(?:5[1-5][0-9]{2}|222[1-9]|22[3-9][0-9]|2[3-6][0-9]{2}|27[01][0-9]|2720)[0-9]{12}",
    )
    mir_pattern: Pattern = re.compile(r"220[0-4][0-9]{12}")
    maestro_pattern: Pattern = re.compile(r"(?:50|5[6-9]|6[0-9])[0-9]{14}")
    amex_pattern: Pattern = re.compile(r"3[4|7][0-9]{13}")
    unionpay_pattern: Pattern = re.compile(r"(?:62|81)[0-9]{14}")

    def test_visa(self, faker, num_samples):
        for _ in range(num_samples):
            number = faker.credit_card_number("visa")
            assert self.visa_pattern.fullmatch(number)

    def test_mastercard(self, faker, num_samples):
        for _ in range(num_samples):
            number = faker.credit_card_number("mastercard")
            assert self.mastercard_pattern.fullmatch(number)

    def test_mir(self, faker, num_samples):
        for _ in range(num_samples):
            number = faker.credit_card_number("mir")
            assert self.mir_pattern.fullmatch(number)

    def test_maestro(self, faker, num_samples):
        for _ in range(num_samples):
            number = faker.credit_card_number("maestro")
            assert self.maestro_pattern.fullmatch(number)

    def test_amex(self, faker, num_samples):
        for _ in range(num_samples):
            number = faker.credit_card_number("amex")
            assert self.amex_pattern.fullmatch(number)

    def test_unionpay(self, faker, num_samples):
        for _ in range(num_samples):
            number = faker.credit_card_number("unionpay")
            assert self.unionpay_pattern.fullmatch(number)

    def test_credit_card_full(self, faker, num_samples):
        for _ in range(num_samples):
            card_data = faker.credit_card_full().split("\n")
            assert re.match("[A-Za-z]+", card_data[1])
            assert card_data[4] in RuRuBankProvider.banks


class TestUkUa:
    mastercard_pattern: Pattern = re.compile(
        r"(?:5[1-5][0-9]{2}|222[1-9]|22[3-9][0-9]|2[3-6][0-9]{2}|27[01][0-9]|2720)[0-9]{12}",
    )
    visa_pattern: Pattern = re.compile(r"4[0-9]{12}([0-9]{3}){0,2}")
    maestro_pattern: Pattern = re.compile(r"(67)[0-9]{14}")
    prostir_pattern: Pattern = re.compile(r"(9)[0-9]{15}")

    def test_mastercard(self, faker, num_samples):
        for _ in range(num_samples):
            number = faker.credit_card_number("mastercard")
            assert self.mastercard_pattern.fullmatch(number)

    def test_visa(self, faker, num_samples):
        for _ in range(num_samples):
            number = faker.credit_card_number("visa")
            assert self.visa_pattern.fullmatch(number)

    def test_maestro(self, faker, num_samples):
        for _ in range(num_samples):
            number = faker.credit_card_number("maestro")
            assert self.maestro_pattern.fullmatch(number)

    def test_prostir(self, faker, num_samples):
        for _ in range(num_samples):
            number = faker.credit_card_number("prostir")
            assert self.prostir_pattern.fullmatch(number)

    def test_credit_card_full(self, faker, num_samples):
        for _ in range(num_samples):
            card_data = faker.credit_card_full("prostir").split("\n")
            assert re.match("[A-Za-z]+", card_data[1])
            assert card_data[4] in UkUaBankProvider.banks
            assert card_data[0] == "ПРОСТІР"


class TestZhCn:
    mastercard_pattern: Pattern = re.compile(
        r"(?:5[1-5][0-9]{2}|222[1-9]|22[3-9][0-9]|2[3-6][0-9]{2}|27[01][0-9]|2720)[0-9]{12}"
    )
    visa_pattern: Pattern = re.compile(r"4[0-9]{12}([0-9]{3}){0,2}")
    unionpay_pattern: Pattern = re.compile(r"62[0-9]{14,17}")  # UnionPay typically starts with 62

    def test_mastercard(self, faker, num_samples):
        for _ in range(num_samples):
            number = faker.credit_card_number("mastercard")
            assert self.mastercard_pattern.fullmatch(number)

    def test_visa(self, faker, num_samples):
        for _ in range(num_samples):
            number = faker.credit_card_number("visa")
            assert self.visa_pattern.fullmatch(number)

    def test_unionpay(self, faker, num_samples):
        for _ in range(num_samples):
            number = faker.credit_card_number("unionpay")
            assert self.unionpay_pattern.fullmatch(number)

    def test_credit_card_full(self, faker, num_samples):
        for _ in range(num_samples):
            card_data = faker.credit_card_full("unionpay").split("\n")
            assert re.match(r"[\u4e00-\u9fff]+", card_data[1])  # Check if owner has Chinese characters
            assert card_data[0] == "UnionPay"  # Ensure correct provider name


class TestCreditCardNumberMask:
    """Test per-position digit mask generation and counting"""

    short_masks = [
        ["4", "9", "27", None, "0"],
        ["0", "0", None, None, "5"],
        [None, "13", None, "7"],
        ["6", "011", None, None, None, "89"],
    ]

    def test_mask_count_matches_exhaustive(self, faker):
        provider = CreditCardProvider(faker)
        for mask in self.short_masks:
            expected = exhaustive_valid_numbers(mask)
            assert provider.credit_card_number_mask_count(mask) == len(expected)

    def test_mask_count_fully_fixed(self, faker):
        provider = CreditCardProvider(faker)
        valid = next(iter(exhaustive_valid_numbers(["4", "9", "27", None, "0"])))
        assert provider.credit_card_number_mask_count(list(valid)) == 1
        invalid = valid[:-1] + str((int(valid[-1]) + 1) % 10)
        assert provider.credit_card_number_mask_count(list(invalid)) == 0

    def test_generated_numbers_cover_exhaustive_set(self, faker):
        provider = CreditCardProvider(faker)
        for mask in self.short_masks:
            expected = exhaustive_valid_numbers(mask)
            samples = {provider.credit_card_number_from_mask(mask) for _ in range(50 * len(expected))}
            assert samples <= expected
            assert samples == expected

    def test_generated_numbers_preserve_fixed_positions_and_leading_zeros(self, faker, num_samples):
        provider = CreditCardProvider(faker)
        mask = ["0", "0", "27", None, None, "5"]
        for _ in range(num_samples):
            number = provider.credit_card_number_from_mask(mask)
            assert len(number) == len(mask)
            assert number.startswith("00")
            assert number[2] in "27"
            assert number.endswith("5")
            assert luhn_valid(number)

    def test_long_mask_count_does_not_enumerate(self, faker):
        provider = CreditCardProvider(faker)
        mask = ["4", "9"] + [None] * 37 + ["0"]
        assert provider.credit_card_number_mask_count(mask) == 10**36
        restricted = ["5", None] * 12 + ["0"]
        assert provider.credit_card_number_mask_count(restricted) == 10**11

    def test_long_mask_generation(self, faker, num_samples):
        provider = CreditCardProvider(faker)
        mask = ["4", "9"] + [None] * 37 + ["0"]
        for _ in range(num_samples):
            number = provider.credit_card_number_from_mask(mask)
            assert len(number) == 40
            assert number.startswith("49")
            assert number.endswith("0")
            assert luhn_valid(number)

    def test_invalid_masks_raise_value_error(self, faker):
        provider = CreditCardProvider(faker)
        for mask in ([], ["4", ""], ["4", "12x"], ["4", 10], ["4", [-1]], ["4", []]):
            with pytest.raises(ValueError):
                provider.credit_card_number_mask_count(mask)

    def test_mask_without_valid_numbers(self, faker):
        provider = CreditCardProvider(faker)
        mask = list("1234567890123453")
        assert provider.credit_card_number_mask_count(mask) == 0
        with pytest.raises(ValueError):
            provider.credit_card_number_from_mask(mask)
