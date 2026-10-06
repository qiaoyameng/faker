import random
from unittest.mock import patch

import pytest

from faker import Faker
from faker.exceptions import UniquenessException
from faker.providers import BaseProvider
from faker.providers.barcode import EanDomain
from faker.utils.unique import DomainSampler, IntegerRangeDomain, PrefixedNumberDomain


class TestIntegerRangeDomain:
    def test_size(self):
        assert IntegerRangeDomain(0, 9999).size == 10000
        assert IntegerRangeDomain(1, 10).size == 10
        assert IntegerRangeDomain(0, 9, step=3).size == 4
        assert IntegerRangeDomain(5, 1).size == 0
        assert IntegerRangeDomain(0, 10**30).size == 10**30 + 1

    def test_rank_unrank_round_trip(self):
        domain = IntegerRangeDomain(-5, 100, step=7)
        for index in range(domain.size):
            assert domain.rank(domain.unrank(index)) == index

    def test_rank_rejects_foreign_values(self):
        domain = IntegerRangeDomain(0, 10, step=2)
        for value in (-2, 1, 12):
            with pytest.raises(ValueError):
                domain.rank(value)

    def test_unrank_rejects_out_of_range_index(self):
        domain = IntegerRangeDomain(0, 10)
        with pytest.raises(IndexError):
            domain.unrank(11)


class TestPrefixedNumberDomain:
    def test_size(self):
        assert PrefixedNumberDomain("PRE-", 3).size == 1000
        assert PrefixedNumberDomain("", 1).size == 10

    def test_rank_unrank_round_trip(self):
        domain = PrefixedNumberDomain("PRE-", 3)
        assert domain.unrank(0) == "PRE-000"
        assert domain.unrank(42) == "PRE-042"
        for index in (0, 1, 999):
            assert domain.rank(domain.unrank(index)) == index

    def test_rank_rejects_foreign_values(self):
        domain = PrefixedNumberDomain("PRE-", 3)
        for value in ("PRE-1", "PRE-0000", "XYZ-000", "PRE-abc"):
            with pytest.raises(ValueError):
                domain.rank(value)


class TestEanDomain:
    def test_size(self):
        assert EanDomain(13).size == 10**12
        assert EanDomain(8, prefix="00").size == 10**5
        assert EanDomain(8, prefix="1234567").size == 1

    def test_rank_unrank_round_trip(self):
        domain = EanDomain(8, prefix="00")
        for index in (0, 1, 12345, domain.size - 1):
            value = domain.unrank(index)
            assert value.startswith("00") and len(value) == 8
            assert domain.rank(value) == index

    def test_rank_rejects_bad_check_digit(self):
        domain = EanDomain(8)
        value = domain.unrank(0)
        bad = value[:-1] + str((int(value[-1]) + 1) % 10)
        with pytest.raises(ValueError):
            domain.rank(bad)


class TestDomainSampler:
    def test_draws_full_domain_without_replacement(self):
        domain = IntegerRangeDomain(1, 100)
        sampler = DomainSampler(domain, random.Random(0))
        values = [sampler.sample() for _ in range(100)]
        assert sorted(values) == list(range(1, 101))
        assert sampler.exhausted
        with pytest.raises(IndexError):
            sampler.sample()

    def test_memory_does_not_scale_with_domain_size(self):
        domain = IntegerRangeDomain(0, 10**9 - 1)
        sampler = DomainSampler(domain, random.Random(0))
        for _ in range(10):
            sampler.sample()
        assert len(sampler._swaps) <= 2 * sampler.drawn


class TestUniqueProxyFiniteDomains:
    def test_small_domain_exhausts_exactly(self):
        """The last value of a small domain is served, and only the next
        draw raises, with no omissions and no premature exhaustion."""
        fake = Faker()
        values = {fake.unique.random_int(min=1, max=10) for _ in range(10)}
        assert values == set(range(1, 11))
        with pytest.raises(UniquenessException, match=r"Exhausted all 10 unique values"):
            fake.unique.random_int(min=1, max=10)

    def test_step_is_respected(self):
        fake = Faker()
        values = {fake.unique.random_int(min=0, max=9, step=3) for _ in range(4)}
        assert values == {0, 3, 6, 9}
        with pytest.raises(UniquenessException):
            fake.unique.random_int(min=0, max=9, step=3)

    def test_argument_combinations_have_independent_state(self):
        fake = Faker()
        for _ in range(3):
            fake.unique.random_int(min=1, max=3)
        with pytest.raises(UniquenessException):
            fake.unique.random_int(min=1, max=3)
        # A different argument combination owns a separate exhaustion state
        assert 1 <= fake.unique.random_int(min=1, max=4) <= 4
        assert 1 <= fake.unique.random_int(min=2, max=3) <= 3

    def test_clear_resets_exhaustion_state(self):
        fake = Faker()
        for _ in range(3):
            fake.unique.random_int(min=1, max=3)
        with pytest.raises(UniquenessException):
            fake.unique.random_int(min=1, max=3)
        fake.unique.clear()
        assert 1 <= fake.unique.random_int(min=1, max=3) <= 3

    def test_large_domain_draws_stay_small_in_memory(self):
        """Drawing a handful of values from a billion-value domain must not
        allocate memory proportional to the domain size."""
        fake = Faker()
        values = {fake.unique.random_int(min=0, max=10**9 - 1) for _ in range(10)}
        assert len(values) == 10
        (state,) = fake.unique._domain_states.values()
        assert len(state._swaps) <= 2 * state.drawn

    def test_domain_path_does_not_call_provider_method(self):
        """Uniqueness on finite domains is not solved by retrying (and thus
        does not depend on the retry attempt limit)."""
        with patch.object(BaseProvider, "random_int", autospec=True) as mock_method:
            fake = Faker()
            values = {fake.unique.random_int(min=1, max=10) for _ in range(10)}
            mock_method.assert_not_called()
        assert values == set(range(1, 11))

    def test_seeded_sampling_is_deterministic(self):
        first = Faker()
        first.seed_instance(1234)
        second = Faker()
        second.seed_instance(1234)
        assert [first.unique.random_int(min=1, max=1000) for _ in range(50)] == [
            second.unique.random_int(min=1, max=1000) for _ in range(50)
        ]

    def test_invalid_range_keeps_original_error(self):
        fake = Faker()
        with pytest.raises(ValueError):
            fake.unique.random_int(min=10, max=1)

    def test_unknown_domain_providers_keep_retry_behavior(self):
        fake = Faker()
        fake.unique.boolean()
        fake.unique.boolean()
        with pytest.raises(UniquenessException, match=r"Got duplicated values after [\d,]+ iterations."):
            fake.unique.boolean()

    def test_excluded_types_bypass_domain_sampling(self):
        fake = Faker()
        proxy = fake.unique.exclude_types([int])
        # ints are excluded from uniqueness enforcement, so duplicates and
        # over-draws are allowed even though the domain only has 2 values
        for _ in range(20):
            assert proxy.random_int(min=1, max=2) in (1, 2)

    def test_ean13_with_fixed_prefix(self):
        fake = Faker()
        codes = {fake.unique.ean13(prefixes=("00",)) for _ in range(100)}
        assert len(codes) == 100
        assert all(code.startswith("00") and len(code) == 13 for code in codes)

    def test_ean8_tiny_domain_exhaustion(self):
        fake = Faker()
        code = fake.unique.ean8(prefixes=((1, 2, 3, 4, 5, 6, 7),))
        assert code.startswith("1234567") and len(code) == 8
        with pytest.raises(UniquenessException, match=r"Exhausted all 1 unique values"):
            fake.unique.ean8(prefixes=((1, 2, 3, 4, 5, 6, 7),))

    def test_ean_without_prefix_uses_full_domain(self):
        fake = Faker()
        codes = {fake.unique.ean8() for _ in range(100)}
        assert len(codes) == 100
        assert all(len(code) == 8 for code in codes)

    def test_ean_with_multiple_prefixes_falls_back_to_retries(self):
        fake = Faker()
        codes = [fake.unique.ean8(prefixes=("00", "11")) for _ in range(10)]
        assert all(code.startswith(("00", "11")) for code in codes)
        assert not fake.unique._domain_states or all(
            state is None for state in fake.unique._domain_states.values()
        )

    def test_en_us_ean13_leading_zero(self):
        fake = Faker("en_US")
        codes = {fake.unique.ean13(leading_zero=True) for _ in range(20)}
        assert len(codes) == 20
        assert all(code.startswith("0") for code in codes)
