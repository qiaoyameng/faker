import itertools

import pytest

from faker.utils.pairwise import COMPLETE, INCOMPLETE, generate_pairwise


# Three small finite fields modelling a test configuration space.
FIELDS = {
    "region": ["cn", "us"],
    "payment": ["alipay", "card"],
    "shipping": ["express", "pickup"],
}

# Forbidden combinations as explicit field-equality conditions.
# Note: ("payment", "alipay") + ("shipping", "express") is not forbidden by
# any single condition, yet no region value makes it legal:
# region="cn" violates the first rule, region="us" violates the second.
CONSTRAINTS = [
    {"region": "cn", "payment": "alipay"},
    {"region": "us", "shipping": "express"},
]

FIELD_NAMES = list(FIELDS)
TRICKY_PAIR = (("payment", "alipay"), ("shipping", "express"))


def is_legal(record):
    return not any(
        all(record[name] == value for name, value in constraint.items()) for constraint in CONSTRAINTS
    )


def all_pairs_of(record):
    return {
        ((first, record[first]), (second, record[second]))
        for first, second in itertools.combinations(FIELD_NAMES, 2)
    }


def brute_force_feasible_pairs():
    feasible = set()
    for values in itertools.product(*(FIELDS[name] for name in FIELD_NAMES)):
        record = dict(zip(FIELD_NAMES, values))
        if is_legal(record):
            feasible |= all_pairs_of(record)
    return feasible


def test_every_record_is_legal():
    result = generate_pairwise(FIELDS, CONSTRAINTS, seed=42)
    assert result.records
    for record in result.records:
        assert set(record) == set(FIELD_NAMES)
        assert all(record[name] in FIELDS[name] for name in FIELD_NAMES)
        assert is_legal(record)


def test_coverage_set_matches_brute_force():
    result = generate_pairwise(FIELDS, CONSTRAINTS, seed=42)
    assert result.status == COMPLETE
    assert result.uncovered_pairs == []
    assert result.undecided_pairs == []
    # Directly check the covered set against an independent brute-force
    # enumeration of every extendable two-field combination.
    assert result.covered_pairs == brute_force_feasible_pairs()
    # The covered set is exactly the union of the pairs of emitted records.
    assert result.covered_pairs == set().union(*(all_pairs_of(r) for r in result.records))


def test_implicitly_impossible_pair_is_not_required():
    result = generate_pairwise(FIELDS, CONSTRAINTS, seed=42)
    feasible = brute_force_feasible_pairs()
    # The tricky pair is genuinely impossible to extend into a legal record.
    assert TRICKY_PAIR not in feasible
    # It must not be covered, but also must not be reported as uncovered or
    # undecided: it is provably impossible, not merely unverified.
    assert TRICKY_PAIR not in result.covered_pairs
    assert TRICKY_PAIR not in result.uncovered_pairs
    assert TRICKY_PAIR not in result.undecided_pairs
    # Explicitly forbidden pairs are excluded the same way.
    assert (("region", "cn"), ("payment", "alipay")) not in result.covered_pairs
    assert (("region", "us"), ("shipping", "express")) not in result.covered_pairs


def test_same_seed_reproduces_records_and_order():
    first = generate_pairwise(FIELDS, CONSTRAINTS, seed=7)
    second = generate_pairwise(FIELDS, CONSTRAINTS, seed=7)
    assert first.records == second.records
    assert first.status == second.status
    assert first.covered_pairs == second.covered_pairs


def test_record_limit_reports_incomplete_with_uncovered_list():
    result = generate_pairwise(FIELDS, CONSTRAINTS, seed=42, max_records=1)
    assert result.status == INCOMPLETE
    assert result.record_limit_reached
    assert len(result.records) == 1
    assert is_legal(result.records[0])
    assert result.uncovered_pairs
    # Uncovered pairs are exactly the feasible pairs not covered so far.
    assert set(result.uncovered_pairs) == brute_force_feasible_pairs() - result.covered_pairs
    assert not (set(result.uncovered_pairs) & result.covered_pairs)


def test_attempt_budget_marks_pairs_undecided_not_impossible():
    result = generate_pairwise(FIELDS, CONSTRAINTS, seed=42, max_attempts=0)
    assert result.status == INCOMPLETE
    assert result.attempt_limit_reached
    assert result.records == []
    assert result.covered_pairs == set()
    # Nothing could be verified, so every pair is undecided -- including
    # pairs that are in fact feasible.  They must not be called impossible.
    assert set(result.undecided_pairs) >= brute_force_feasible_pairs()
    assert TRICKY_PAIR in result.undecided_pairs


def test_unknown_field_in_constraint_raises():
    with pytest.raises(ValueError):
        generate_pairwise(FIELDS, [{"country": "cn"}])


def test_empty_candidate_values_raise():
    with pytest.raises(ValueError):
        generate_pairwise({"region": []})


def test_unconstrained_space_covers_full_cartesian_pairs():
    fields = {"a": [1, 2, 3], "b": ["x", "y"], "c": [True, False]}
    result = generate_pairwise(fields, seed=1)
    assert result.status == COMPLETE
    expected = {
        ((f1, v1), (f2, v2))
        for f1, f2 in itertools.combinations(fields, 2)
        for v1 in fields[f1]
        for v2 in fields[f2]
    }
    assert result.covered_pairs == expected
