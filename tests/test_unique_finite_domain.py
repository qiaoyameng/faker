"""Tests for the finite-domain unique sampling path.

Providers that declare a finite value domain are sampled without replacement
through a rank/unrank based permutation.  This must:

- never report exhaustion before the very last value has been drawn;
- cover every value exactly once (no omissions, no duplicates);
- keep independent exhaustion state per argument combination;
- be reset by ``clear()``;
- use memory proportional to the number of draws, not to the domain size.
"""

import sys

import pytest

from faker import Faker
from faker.exceptions import UniquenessException
from faker.providers.domains import (
    IntegerRangeDomain,
    NumericCodeDomain,
    ValueDomain,
    finite_domain,
)
from faker.providers import BaseProvider


class TestIntegerRangeDomain:
    def test_size_matches_range_object(self):
        for args, expected in [
            ((0, 9), range(0, 10)),
            ((0, 15, 3), range(0, 16, 3)),
            ((5, 5), range(5, 6)),
            ((10, 5), range(10, 6)),
            ((10, 2, -2), range(10, 1, -2)),
        ]:
            domain = IntegerRangeDomain(*args)
            assert domain.size == len(expected)
            assert [domain.unrank(i) for i in range(domain.size)] == list(expected)
            for index, value in enumerate(expected):
                assert domain.rank(value) == index

    def test_invalid_step(self):
        with pytest.raises(ValueError):
            IntegerRangeDomain(0, 10, 0)

    def test_rank_rejects_outside_values(self):
        domain = IntegerRangeDomain(0, 10, 2)
        with pytest.raises(ValueError):
            domain.rank(1)
        with pytest.raises(ValueError):
            domain.rank(20)

    def test_unrank_bounds(self):
        domain = IntegerRangeDomain(0, 4)
        with pytest.raises(IndexError):
            domain.unrank(5)
        with pytest.raises(IndexError):
            domain.unrank(-1)


class TestNumericCodeDomain:
    def test_size_and_unrank(self):
        domain = NumericCodeDomain(prefix="SKU-", digits=3)
        assert domain.size == 1000
        assert domain.unrank(0) == "SKU-000"
        assert domain.unrank(42) == "SKU-042"
        assert domain.unrank(999) == "SKU-999"

    def test_rank_roundtrip(self):
        domain = NumericCodeDomain(prefix="SKU-", digits=3)
        for index in (0, 1, 42, 999):
            assert domain.rank(domain.unrank(index)) == index

    def test_zero_digits(self):
        domain = NumericCodeDomain(prefix="X", digits=0)
        assert domain.size == 1
        assert domain.unrank(0) == "X"
        assert domain.rank("X") == 0

    def test_invalid_arguments(self):
        with pytest.raises(TypeError):
            NumericCodeDomain(prefix=123, digits=3)
        with pytest.raises(ValueError):
            NumericCodeDomain(prefix="", digits=-1)
        with pytest.raises(ValueError):
            NumericCodeDomain(prefix="", digits=True)

    def test_rank_rejects_malformed(self):
        domain = NumericCodeDomain(prefix="SKU-", digits=3)
        for bad in ("XXX-001", "SKU-1", "SKU-0001", 1, None, "SKU-abc"):
            with pytest.raises(ValueError):
                domain.rank(bad)


class TestUniqueFiniteSampling:
    def test_small_range_exhausts_exactly_at_last_value(self):
        fake = Faker()
        values = [fake.unique.random_int(min=1, max=5) for _ in range(5)]
        assert sorted(values) == [1, 2, 3, 4, 5]
        # The next draw reports exhaustion; it is never reported early.
        with pytest.raises(UniquenessException, match="exhausting the finite domain"):
            fake.unique.random_int(min=1, max=5)

    def test_step_range_full_coverage(self):
        fake = Faker()
        expected = set(range(0, 16, 3))
        seen = {fake.unique.random_int(min=0, max=15, step=3) for _ in range(len(expected))}
        assert seen == expected
        with pytest.raises(UniquenessException):
            fake.unique.random_int(min=0, max=15, step=3)

    def test_clear_resets_exhaustion(self):
        fake = Faker()
        for _ in range(5):
            fake.unique.random_int(min=1, max=5)
        with pytest.raises(UniquenessException):
            fake.unique.random_int(min=1, max=5)
        fake.unique.clear()
        value = fake.unique.random_int(min=1, max=5)
        assert value in {1, 2, 3, 4, 5}

    def test_state_is_per_argument_combination(self):
        fake = Faker()
        for _ in range(5):
            fake.unique.random_int(min=1, max=5)
        # A different signature has its own, non-exhausted pool.
        fake.unique.random_int(min=1, max=6)
        fake.unique.random_int(min=2, max=5)

    def test_numeric_code_full_coverage(self):
        fake = Faker()
        codes = {fake.unique.numeric_code(prefix="X", digits=2) for _ in range(100)}
        assert codes == {f"X{i:02d}" for i in range(100)}
        with pytest.raises(UniquenessException):
            fake.unique.numeric_code(prefix="X", digits=2)

    def test_numeric_code_keeps_leading_zeroes_and_prefix(self):
        fake = Faker()
        for _ in range(50):
            code = fake.unique.numeric_code(prefix="ID-", digits=6)
            assert code.startswith("ID-") and len(code) == len("ID-") + 6

    def test_numeric_code_signatures_are_independent(self):
        fake = Faker()
        for _ in range(10):
            fake.unique.numeric_code(prefix="A", digits=1)
        fake.unique.numeric_code(prefix="B", digits=1)
        fake.unique.numeric_code(prefix="A", digits=2)

    def test_memory_does_not_grow_with_domain_size(self):
        import tracemalloc

        fake = Faker()
        tracemalloc.start()
        before = tracemalloc.take_snapshot()
        draws = [fake.unique.random_int(min=0, max=1_000_000_000 - 1) for _ in range(10)]
        after = tracemalloc.take_snapshot()
        assert len(set(draws)) == 10
        allocated_kb = sum(stat.size_diff for stat in after.compare_to(before, "lineno")) / 1024
        assert allocated_kb < 200

    def test_huge_numeric_code_domain_uses_bounded_memory(self):
        fake = Faker()
        codes = [fake.unique.numeric_code(prefix="", digits=10) for _ in range(10)]
        assert len(set(codes)) == 10

    def test_unknown_domain_provider_keeps_retry_behavior(self):
        # ``boolean`` declares no finite domain, so it relies on retries and
        # raises after exhausting its two possible values on the third draw.
        fake = Faker()
        fake.unique.boolean()
        fake.unique.boolean()
        with pytest.raises(UniquenessException, match=r"Got duplicated values after [\d,]+ iterations."):
            fake.unique.boolean()

    def test_shared_state_across_locale_proxies(self):
        fake = Faker(["en_US", "fr_FR"])
        seen = set()
        for _ in range(5):
            seen.add(fake.unique["en_US"].random_int(min=1, max=10))
        for _ in range(5):
            seen.add(fake.unique["fr_FR"].random_int(min=1, max=10))
        assert len(seen) == 10
        with pytest.raises(UniquenessException):
            fake.unique["en_US"].random_int(min=1, max=10)

    def test_custom_provider_domain(self):
        class EvenDomain(ValueDomain):
            def __init__(self, count):
                self._size = count

            @property
            def size(self):
                return self._size

            def unrank(self, index):
                return index * 2

            def rank(self, value):
                return value // 2

        class CustomProvider(BaseProvider):
            @finite_domain(EvenDomain)
            def evens(self, count=5):
                domain = EvenDomain(count)
                return domain.unrank(self.generator.random.randrange(domain.size))

        fake = Faker()
        fake.add_provider(CustomProvider)
        values = [fake.unique.evens(count=4) for _ in range(4)]
        assert sorted(values) == [0, 2, 4, 6]
        with pytest.raises(UniquenessException):
            fake.unique.evens(count=4)

    def test_seeded_sequence_is_reproducible(self):
        fake_a = Faker()
        fake_a.seed_instance(4242)
        first = [fake_a.unique.random_int(min=0, max=99) for _ in range(10)]

        fake_b = Faker()
        fake_b.seed_instance(4242)
        second = [fake_b.unique.random_int(min=0, max=99) for _ in range(10)]
        assert first == second
