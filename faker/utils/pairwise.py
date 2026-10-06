"""Pairwise (all-pairs) test data generation over finite discrete fields.

Given a set of fields with finite candidate values and a list of forbidden
combinations (each described as explicit field-equality conditions), this
module generates a list of records such that:

* every generated record is legal (violates no forbidden combination), and
* every two-field value combination that can be extended into at least one
  legal full record appears in at least one generated record.

The number of records is not guaranteed to be minimal.  When a resource
limit (``max_records`` / ``max_attempts``) is hit before full coverage is
reached, the result carries an explicit ``"incomplete"`` status together
with the list of uncovered pairs.  Pairs whose feasibility could not be
decided within the attempt budget are reported as *undecided* and are never
silently treated as impossible.

Generation is deterministic: the same ``seed`` yields the same records in
the same order.
"""

import itertools

from dataclasses import dataclass, field
from random import Random
from typing import Any, Dict, Hashable, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

#: A canonical two-field combination: ((field_a, value_a), (field_b, value_b))
#: with field_a ordered before field_b according to the field declaration order.
Pair = Tuple[Tuple[str, Hashable], Tuple[str, Hashable]]

#: Search outcome of a witness search for a partial assignment.
_FOUND = "found"
_IMPOSSIBLE = "impossible"
_UNKNOWN = "unknown"

COMPLETE = "complete"
INCOMPLETE = "incomplete"


@dataclass
class PairwiseResult:
    """Outcome of :func:`generate_pairwise`."""

    status: str
    records: List[Dict[str, Hashable]]
    covered_pairs: Set[Pair] = field(default_factory=set)
    uncovered_pairs: List[Pair] = field(default_factory=list)
    undecided_pairs: List[Pair] = field(default_factory=list)
    record_limit_reached: bool = False
    attempt_limit_reached: bool = False

    @property
    def complete(self) -> bool:
        return self.status == COMPLETE


class _AttemptBudget:
    """Shared countdown of remaining search steps."""

    def __init__(self, limit: int) -> None:
        self.remaining = max(0, limit)

    def consume(self) -> bool:
        """Use one step; return False when the budget is exhausted."""
        if self.remaining <= 0:
            return False
        self.remaining -= 1
        return True


def _normalize_constraints(
    constraints: Iterable[Mapping[str, Hashable]],
    field_names: Sequence[str],
) -> List[Tuple[Tuple[str, Hashable], ...]]:
    known = set(field_names)
    normalized = []
    for constraint in constraints:
        items = tuple(constraint.items())
        for name, _ in items:
            if name not in known:
                raise ValueError(f"forbidden combination references unknown field: {name!r}")
        if not items:
            raise ValueError("forbidden combinations must contain at least one field condition")
        normalized.append(items)
    return normalized


def _is_legal_partial(
    assignment: Mapping[str, Hashable],
    constraints: Sequence[Tuple[Tuple[str, Hashable], ...]],
) -> bool:
    """A partial assignment is legal when no forbidden combination fully matches it."""
    for constraint in constraints:
        if all(name in assignment and assignment[name] == value for name, value in constraint):
            return False
    return True


def _pairs_of(record: Mapping[str, Hashable], field_names: Sequence[str]) -> Set[Pair]:
    pairs = set()
    for first, second in itertools.combinations(field_names, 2):
        pairs.add(((first, record[first]), (second, record[second])))
    return pairs


def _find_witness(
    assignment: Dict[str, Hashable],
    remaining_fields: Sequence[str],
    values_by_field: Mapping[str, Sequence[Hashable]],
    constraints: Sequence[Tuple[Tuple[str, Hashable], ...]],
    budget: _AttemptBudget,
) -> Tuple[str, Optional[Dict[str, Hashable]]]:
    """Try to extend ``assignment`` into a legal full record.

    Returns ``(_FOUND, record)`` when a legal extension exists,
    ``(_IMPOSSIBLE, None)`` when the whole remaining search space was
    explored without finding one, and ``(_UNKNOWN, None)`` when the attempt
    budget ran out before either could be decided.
    """
    if not remaining_fields:
        return (_FOUND, dict(assignment)) if _is_legal_partial(assignment, constraints) else (_IMPOSSIBLE, None)
    name = remaining_fields[0]
    rest = remaining_fields[1:]
    exhausted = True
    for value in values_by_field[name]:
        if not budget.consume():
            return _UNKNOWN, None
        assignment[name] = value
        if _is_legal_partial(assignment, constraints):
            outcome, witness = _find_witness(assignment, rest, values_by_field, constraints, budget)
            if outcome == _FOUND:
                del assignment[name]
                return _FOUND, witness
            if outcome == _UNKNOWN:
                exhausted = False
        del assignment[name]
    return (_IMPOSSIBLE if exhausted else _UNKNOWN), None


def generate_pairwise(
    fields: Mapping[str, Sequence[Hashable]],
    constraints: Iterable[Mapping[str, Hashable]] = (),
    *,
    seed: Optional[int] = 0,
    max_records: int = 1000,
    max_attempts: int = 100000,
) -> PairwiseResult:
    """Generate a pairwise-covering set of legal records.

    :param fields: mapping of field name to its finite candidate values.
    :param constraints: forbidden combinations; each is a mapping of field
        name to a forbidden value, and a record is forbidden when it matches
        every condition of any single combination.
    :param seed: seed controlling search/tie-break order; identical seeds
        reproduce identical records and ordering.
    :param max_records: resource limit on the number of emitted records.
    :param max_attempts: resource limit on total feasibility-search steps.
    :return: a :class:`PairwiseResult`; ``status`` is ``"complete"`` only
        when every extendable pair is covered and nothing is undecided.
    """
    field_names = list(fields)
    values_by_field: Dict[str, List[Hashable]] = {}
    for name, values in fields.items():
        deduped = list(dict.fromkeys(values))
        if not deduped:
            raise ValueError(f"field {name!r} must have at least one candidate value")
        values_by_field[name] = deduped
    normalized = _normalize_constraints(constraints, field_names)
    rng = Random(seed)
    budget = _AttemptBudget(max_attempts)

    # Phase 1: classify every two-field combination and collect a witness
    # (a legal full record extending it) for each feasible pair.
    all_pairs: List[Pair] = []
    for first, second in itertools.combinations(field_names, 2):
        for value_first in values_by_field[first]:
            for value_second in values_by_field[second]:
                all_pairs.append(((first, value_first), (second, value_second)))
    rng.shuffle(all_pairs)

    witnesses: Dict[Pair, Dict[str, Hashable]] = {}
    feasible: Set[Pair] = set()
    undecided: Set[Pair] = set()
    for pair in all_pairs:
        assignment = {pair[0][0]: pair[0][1], pair[1][0]: pair[1][1]}
        remaining = [name for name in field_names if name not in assignment]
        outcome, witness = _find_witness(assignment, remaining, values_by_field, normalized, budget)
        if outcome == _FOUND:
            assert witness is not None
            feasible.add(pair)
            witnesses[pair] = witness
        elif outcome == _UNKNOWN:
            undecided.add(pair)

    # Phase 2: greedily pick witness records covering the most still
    # uncovered pairs.  Every picked record covers at least its own pair,
    # so the loop always makes progress while feasible pairs remain.
    uncovered: Set[Pair] = set(feasible) | set(undecided)
    covered: Set[Pair] = set()
    records: List[Dict[str, Hashable]] = []
    record_limit_reached = False
    while uncovered:
        if len(records) >= max_records:
            record_limit_reached = True
            break
        best_record: Optional[Dict[str, Hashable]] = None
        best_gain = 0
        for pair in all_pairs:
            if pair not in uncovered or pair not in witnesses:
                continue
            gain = len(_pairs_of(witnesses[pair], field_names) & uncovered)
            if gain > best_gain:
                best_gain = gain
                best_record = witnesses[pair]
        if best_record is None:
            break  # only undecided pairs without witnesses remain
        records.append(best_record)
        newly_covered = _pairs_of(best_record, field_names) & uncovered
        covered |= newly_covered
        uncovered -= newly_covered

    sort_key = lambda pair: (pair[0][0], repr(pair[0][1]), pair[1][0], repr(pair[1][1]))  # noqa: E731
    uncovered_feasible = sorted(uncovered & feasible, key=sort_key)
    still_undecided = sorted(uncovered & undecided, key=sort_key)
    status = COMPLETE if not uncovered else INCOMPLETE
    return PairwiseResult(
        status=status,
        records=records,
        covered_pairs=covered,
        uncovered_pairs=uncovered_feasible,
        undecided_pairs=still_undecided,
        record_limit_reached=record_limit_reached,
        attempt_limit_reached=budget.remaining <= 0,
    )
