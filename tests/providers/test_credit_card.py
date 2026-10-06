import re
from collections import Counter
from itertools import product
from typing import Pattern

import pytest

from faker.providers.bank.ru_RU import Provider as RuRuBankProvider
from faker.providers.bank.uk_UA import Provider as UkUaBankProvider
from faker.providers.credit_card import Provider as CreditCardProvider


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

    def test_credit_card_number_from_mask_matches_exhaustive_set(self, faker):
        digit_sets = [0, 0, "27", range(10), range(10), {0}]
        expected = {
            number
            for number in (
                "".join(digits)
                for digits in product("0", "0", "27", "0123456789", "0123456789", "0")
            )
            if self.luhn_is_valid(number)
        }

        assert faker.credit_card_number_count(digit_sets) == len(expected)
        assert faker.credit_card_number_mask_count(digit_sets) == len(expected)
        assert faker.credit_card_number_mask_count("####") == 1000

        faker.seed_instance(1234)
        generated = {
            faker.credit_card_number_from_mask(digit_sets)
            for _ in range(500)
        }

        assert generated == expected
        for number in generated:
            assert number.startswith("00")
            assert number.endswith("0")
            assert number[2] in "27"

    def test_credit_card_number_from_mask_uniformly_samples_valid_numbers(self, faker):
        digit_sets = ["12", "27", range(10), range(10)]
        expected = {
            "".join(digits)
            for digits in product("12", "27", "0123456789", "0123456789")
            if self.luhn_is_valid("".join(digits))
        }

        assert faker.credit_card_number_count(digit_sets) == len(expected)
        faker.seed_instance(1234)
        counts = Counter(
            faker.credit_card_number_from_digit_sets(digit_sets)
            for _ in range(4000)
        )

        assert set(counts) == expected
        assert all(self.luhn_is_valid(number) for number in counts)
        assert min(counts.values()) > 75
        assert max(counts.values()) < 125

    def test_credit_card_number_from_mask_respects_fixed_check_digit(self, faker):
        valid_mask = ["12", "27", range(10), range(10), "0"]
        invalid_mask = ["1", "2", "3"]

        for _ in range(20):
            number = faker.credit_card_number_from_mask(valid_mask)
            assert number.endswith("0")
            assert self.luhn_is_valid(number)

        assert faker.credit_card_number_count(valid_mask) > 0
        assert faker.credit_card_number_count(invalid_mask) == 0
        with pytest.raises(ValueError, match="No credit card numbers"):
            faker.credit_card_number_from_mask(invalid_mask)

    def test_credit_card_number_from_mask_counts_long_mask_without_enumeration(self, faker):
        digit_sets = [0, 0, "27", *[range(10) for _ in range(996)], {0}]

        assert faker.credit_card_number_count(digit_sets) == 2 * 10**995

        faker.seed_instance(1234)
        number = faker.credit_card_number_from_mask(digit_sets)

        assert len(number) == 1000
        assert number.startswith("00")
        assert number.endswith("0")
        assert number[2] in "27"
        assert self.luhn_is_valid(number)

    @pytest.mark.parametrize(
        "digit_sets",
        [
            [],
            [range(10), []],
            [range(10), (10,)],
            [range(10), "a"],
        ],
    )
    def test_credit_card_number_from_mask_rejects_invalid_masks(self, faker, digit_sets):
        with pytest.raises((TypeError, ValueError)):
            faker.credit_card_number_count(digit_sets)

    @staticmethod
    def luhn_is_valid(number: str) -> bool:
        total = 0
        for index, digit in enumerate(reversed(number)):
            value = int(digit)
            if index % 2 == 1:
                value *= 2
                if value > 9:
                    value -= 9
            total += value
        return total % 10 == 0


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
