import itertools
import unittest

from faker.utils.pairwise import generate_pairwise

# Three small finite field enumerations, e.g. a test configuration space of
# region / payment method / delivery type.
FIELDS = {
    "region": ["cn", "us", "eu"],
    "payment": ["card", "cod", "wallet"],
    "delivery": ["express", "pickup", "freight"],
}

# Forbidden combinations as explicit field-equality conditions. Note that no
# constraint mentions ("payment", "wallet") together with ("delivery",
# "pickup"): that pair is only *indirectly* impossible, because every region
# rules it out through a different explicit constraint.
FORBIDDEN = [
    {"payment": "cod", "delivery": "freight"},
    {"region": "cn", "payment": "wallet"},
    {"region": "us", "delivery": "pickup"},
    {"region": "eu", "payment": "wallet"},
]

IMPLICITLY_INFEASIBLE_PAIR = (("payment", "wallet"), ("delivery", "pickup"))


def is_valid(record):
    return not any(all(record[field] == value for field, value in constraint.items()) for constraint in FORBIDDEN)


def pairs_of(record):
    names = list(FIELDS)
    return {
        ((first, record[first]), (second, record[second]))
        for i, first in enumerate(names)
        for second in names[i + 1 :]
    }


def expected_feasible_pairs():
    feasible = set()
    for values in itertools.product(*FIELDS.values()):
        record = dict(zip(FIELDS, values))
        if is_valid(record):
            feasible |= pairs_of(record)
    return feasible


class TestPairwiseGeneration(unittest.TestCase):
    def test_every_record_is_valid(self):
        result = generate_pairwise(FIELDS, FORBIDDEN, seed=42)
        self.assertTrue(result.records)
        for record in result.records:
            self.assertEqual(set(record), set(FIELDS))
            self.assertTrue(is_valid(record), record)

    def test_covers_exactly_the_feasible_pair_set(self):
        result = generate_pairwise(FIELDS, FORBIDDEN, seed=42)
        self.assertEqual(result.status, "complete")
        self.assertTrue(result.complete)
        self.assertEqual(result.uncovered_pairs, [])
        self.assertEqual(result.undetermined_pairs, [])
        # Directly check the covered set against a brute-force reference.
        self.assertEqual(result.covered_pairs, expected_feasible_pairs())
        # The covered set must be realized by the actual records.
        covered_by_records = set().union(*(pairs_of(record) for record in result.records))
        self.assertEqual(covered_by_records, result.covered_pairs)

    def test_indirectly_impossible_pair_is_reported_infeasible(self):
        result = generate_pairwise(FIELDS, FORBIDDEN, seed=42)
        # Not explicitly forbidden, but cannot coexist with any region value.
        self.assertIn(IMPLICITLY_INFEASIBLE_PAIR, result.infeasible_pairs)
        self.assertNotIn(IMPLICITLY_INFEASIBLE_PAIR, result.covered_pairs)
        # Every explicitly forbidden two-field combination is infeasible too.
        self.assertIn((("payment", "cod"), ("delivery", "freight")), result.infeasible_pairs)
        self.assertIn((("region", "cn"), ("payment", "wallet")), result.infeasible_pairs)
        self.assertIn((("region", "us"), ("delivery", "pickup")), result.infeasible_pairs)
        # Nothing feasible is misclassified as infeasible.
        self.assertTrue(set(result.infeasible_pairs).isdisjoint(expected_feasible_pairs()))

    def test_same_seed_reproduces_records_and_order(self):
        first = generate_pairwise(FIELDS, FORBIDDEN, seed=7)
        second = generate_pairwise(FIELDS, FORBIDDEN, seed=7)
        self.assertEqual(first.records, second.records)
        self.assertEqual(first.covered_pairs, second.covered_pairs)

    def test_record_limit_yields_incomplete_status_with_uncovered_list(self):
        result = generate_pairwise(FIELDS, FORBIDDEN, seed=42, max_records=1)
        self.assertEqual(len(result.records), 1)
        self.assertFalse(result.complete)
        self.assertEqual(result.status, "incomplete")
        self.assertTrue(result.uncovered_pairs)
        # Uncovered pairs are feasible ones the records genuinely miss.
        self.assertTrue(set(result.uncovered_pairs) <= expected_feasible_pairs())
        self.assertTrue(set(result.uncovered_pairs).isdisjoint(pairs_of(result.records[0])))

    def test_search_budget_exhaustion_is_undetermined_not_infeasible(self):
        result = generate_pairwise(FIELDS, FORBIDDEN, seed=42, max_search_steps=0)
        self.assertEqual(result.records, [])
        self.assertFalse(result.complete)
        self.assertEqual(result.status, "incomplete")
        # Nothing could be decided, so nothing is claimed impossible.
        self.assertEqual(result.infeasible_pairs, [])
        self.assertEqual(len(result.undetermined_pairs), 27)

    def test_unknown_constraint_field_is_rejected(self):
        with self.assertRaises(ValueError):
            generate_pairwise(FIELDS, [{"unknown": "x"}])


if __name__ == "__main__":
    unittest.main()
