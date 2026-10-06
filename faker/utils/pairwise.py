"""
Pairwise (2-way) combinatorial test data generation with explicit constraints.

Fields have finite candidate values. Forbidden combinations are described as
explicit field-equality conditions: each constraint is a mapping of field
names to values, and a record is invalid when it matches *all* equalities of
any single constraint.

The generator guarantees that every produced record is valid, and that every
two-field value combination that can be extended into at least one valid
complete record appears in at least one produced record. Combinations that
cannot be extended (including ones not explicitly forbidden but ruled out
indirectly through other fields) are reported separately. When a resource
limit is hit, the result is explicitly marked incomplete and combinations
that could not be decided are reported as undetermined rather than treated
as impossible.
"""

import random

from dataclasses import dataclass
from typing import Any, Dict, Hashable, List, Mapping, Optional, Sequence, Set, Tuple

Pair = Tuple[Tuple[str, Hashable], Tuple[str, Hashable]]

__all__ = [
    "Pair",
    "PairwiseResult",
    "generate_pairwise",
]


class _SearchBudgetExceeded(Exception):
    """Raised internally when a feasibility search exceeds its step budget."""


@dataclass
class PairwiseResult:
    """Outcome of :func:`generate_pairwise`.

    ``records``
        Produced records in generation order. Every record satisfies all
        constraints.
    ``covered_pairs``
        Two-field combinations covered by at least one record.
    ``uncovered_pairs``
        Combinations proven feasible but not covered because ``max_records``
        was reached. Empty unless the result is incomplete.
    ``infeasible_pairs``
        Combinations proven to have no valid complete record. This includes
        combinations that are not explicitly forbidden but cannot coexist
        with any value of the remaining fields.
    ``undetermined_pairs``
        Combinations whose feasibility could not be decided within
        ``max_search_steps``. They are never reported as infeasible.
    """

    records: List[Dict[str, Hashable]]
    covered_pairs: Set[Pair]
    uncovered_pairs: List[Pair]
    infeasible_pairs: List[Pair]
    undetermined_pairs: List[Pair]

    @property
    def complete(self) -> bool:
        return not self.uncovered_pairs and not self.undetermined_pairs

    @property
    def status(self) -> str:
        return "complete" if self.complete else "incomplete"


def _violates(assignment: Mapping[str, Hashable], constraints: Sequence[Mapping[str, Hashable]]) -> bool:
    for constraint in constraints:
        if all(field in assignment and assignment[field] == value for field, value in constraint.items()):
            return True
    return False


def _search(
    names: Sequence[str],
    values: Mapping[str, Sequence[Hashable]],
    constraints: Sequence[Mapping[str, Hashable]],
    assignment: Dict[str, Hashable],
    steps: List[int],
    max_steps: int,
) -> Optional[Dict[str, Hashable]]:
    steps[0] += 1
    if steps[0] > max_steps:
        raise _SearchBudgetExceeded
    if len(assignment) == len(names):
        return dict(assignment)
    name = next(name for name in names if name not in assignment)
    for value in values[name]:
        assignment[name] = value
        if not _violates(assignment, constraints):
            result = _search(names, values, constraints, assignment, steps, max_steps)
            if result is not None:
                del assignment[name]
                return result
        del assignment[name]
    return None


def _solve(
    names: Sequence[str],
    values: Mapping[str, Sequence[Hashable]],
    constraints: Sequence[Mapping[str, Hashable]],
    assignment: Optional[Dict[str, Hashable]] = None,
    max_steps: int = 10000,
) -> Optional[Dict[str, Hashable]]:
    return _search(names, values, constraints, dict(assignment or {}), [0], max_steps)


def _random_record(
    names: Sequence[str],
    values: Mapping[str, Sequence[Hashable]],
    constraints: Sequence[Mapping[str, Hashable]],
    rng: random.Random,
    max_steps: int,
) -> Optional[Dict[str, Hashable]]:
    shuffled = {name: rng.sample(list(values[name]), len(values[name])) for name in names}
    return _solve(names, shuffled, constraints, max_steps=max_steps)


def _pairs_of(record: Mapping[str, Hashable], names: Sequence[str]) -> Set[Pair]:
    pairs: Set[Pair] = set()
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            pairs.add(((first, record[first]), (second, record[second])))
    return pairs


def generate_pairwise(
    fields: Mapping[str, Sequence[Hashable]],
    forbidden: Sequence[Mapping[str, Hashable]] = (),
    seed: Optional[int] = None,
    max_records: Optional[int] = None,
    max_search_steps: int = 10000,
    candidate_attempts: int = 8,
) -> PairwiseResult:
    """Generate a pairwise (2-way) covering set of valid records.

    :param fields: Mapping of field name to its finite candidate values.
        Iteration order of the mapping defines the canonical field order.
    :param forbidden: Constraints described as explicit field-equality
        conditions. A record matching every equality of one constraint is
        invalid and never produced.
    :param seed: Seed for the internal RNG. The same seed, fields, and
        constraints always produce the same records in the same order.
    :param max_records: Optional cap on the number of produced records. If
        the cap is reached first, feasible-but-uncovered combinations are
        reported in ``uncovered_pairs`` and the result is incomplete.
    :param max_search_steps: Per-search budget of solver steps. If exceeded,
        the affected combinations are reported in ``undetermined_pairs`` and
        the result is incomplete; they are never treated as infeasible.
    :param candidate_attempts: Number of random candidate records tried per
        greedy round in addition to the guaranteed witness record.
    """
    names = list(fields)
    values: Dict[str, List[Hashable]] = {name: list(fields[name]) for name in names}
    for name, candidates in values.items():
        if not candidates:
            raise ValueError(f"field {name!r} must have at least one candidate value")
    constraints = [dict(constraint) for constraint in forbidden]
    for constraint in constraints:
        for name in constraint:
            if name not in values:
                raise ValueError(f"constraint references unknown field {name!r}")

    value_indexes: Dict[str, Dict[Hashable, int]] = {
        name: {value: index for index, value in enumerate(candidates)} for name, candidates in values.items()
    }
    name_indexes = {name: index for index, name in enumerate(names)}

    def pair_key(pair: Pair) -> Tuple[int, int, int, int]:
        (first_name, first_value), (second_name, second_value) = pair
        return (
            name_indexes[first_name],
            value_indexes[first_name][first_value],
            name_indexes[second_name],
            value_indexes[second_name][second_value],
        )

    all_pairs: List[Pair] = []
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            for first_value in values[first]:
                for second_value in values[second]:
                    all_pairs.append(((first, first_value), (second, second_value)))

    witnesses: Dict[Pair, Dict[str, Hashable]] = {}
    infeasible: List[Pair] = []
    undetermined: List[Pair] = []
    for pair in all_pairs:
        (first_name, first_value), (second_name, second_value) = pair
        partial = {first_name: first_value, second_name: second_value}
        try:
            witness = _solve(names, values, constraints, partial, max_search_steps)
        except _SearchBudgetExceeded:
            undetermined.append(pair)
            continue
        if witness is None:
            infeasible.append(pair)
        else:
            witnesses[pair] = witness

    rng = random.Random(seed)
    records: List[Dict[str, Hashable]] = []
    covered: Set[Pair] = set()
    uncovered: Set[Pair] = set(witnesses)
    while uncovered:
        if max_records is not None and len(records) >= max_records:
            break
        anchor = min(uncovered, key=pair_key)
        candidates = [witnesses[anchor]]
        for _ in range(candidate_attempts):
            try:
                candidate = _random_record(names, values, constraints, rng, max_search_steps)
            except _SearchBudgetExceeded:
                continue
            if candidate is not None:
                candidates.append(candidate)
        best = max(candidates, key=lambda record: len(_pairs_of(record, names) & uncovered))
        newly_covered = _pairs_of(best, names) & uncovered
        records.append(best)
        covered |= newly_covered
        uncovered -= newly_covered

    return PairwiseResult(
        records=records,
        covered_pairs=covered,
        uncovered_pairs=sorted(uncovered, key=pair_key),
        infeasible_pairs=sorted(infeasible, key=pair_key),
        undetermined_pairs=sorted(undetermined, key=pair_key),
    )
