from __future__ import annotations

from collections import OrderedDict
from typing import Any, Callable, Hashable, Iterator, Mapping, Sequence

from .exceptions import InfeasibleQuotaException

QuotaMapping = Mapping[Hashable, int]
JointCounts = dict[Hashable, dict[Hashable, int]]
ExtraFields = Mapping[str, str | Callable[[], Any]]


class QuotaRecordIterator:
    """Lazily emit a uniformly shuffled multiset for fixed joint counts."""

    def __init__(
        self,
        counts: JointCounts,
        columns: Sequence[Hashable],
        total: int,
        random: Any,
        generator: Any,
        row_key: str,
        column_key: str,
        fields: Mapping[str, str | Callable[[], Any]],
    ) -> None:
        self.joint_counts = counts
        self._remaining = [[counts[row].get(column, 0) for column in columns] for row in counts]
        self._rows = list(counts)
        self._columns = [list(columns) for _ in counts]
        self._total = total
        self._random = random
        self._generator = generator
        self._row_key = row_key
        self._column_key = column_key
        self._fields = fields

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return self

    def __next__(self) -> dict[str, Any]:
        """Return the next quota record without materializing the full result."""
        if self._total == 0:
            raise StopIteration

        position = self._random.randrange(self._total)
        cumulative = 0
        selected_row = 0
        selected_column = 0

        for row_index, remaining_row in enumerate(self._remaining):
            for column_index, count in enumerate(remaining_row):
                cumulative += count
                if position < cumulative:
                    selected_row = row_index
                    selected_column = column_index
                    break
            if position < cumulative:
                break

        row = self._rows[selected_row]
        column = self._columns[selected_row][selected_column]
        self._remaining[selected_row][selected_column] -= 1
        self._total -= 1

        record = {self._row_key: row, self._column_key: column}
        for name, provider in self._fields.items():
            if isinstance(provider, str):
                record[name] = self._generator.format(provider)
            else:
                record[name] = provider()
        return record

    def __length_hint__(self) -> int:
        return self._total


def dual_quota_records(
    generator: Any,
    row_quotas: QuotaMapping,
    column_quotas: QuotaMapping,
    forbidden: Any | None = (),
    *,
    row_key: str = "row",
    column_key: str = "column",
    fields: ExtraFields | None = None,
) -> tuple[JointCounts, QuotaRecordIterator]:
    """Return exact joint counts and a lazily shuffled record iterator."""

    rows = _normalize_quotas(row_quotas, "row_quotas")
    columns = _normalize_quotas(column_quotas, "column_quotas")
    forbidden_pairs = _normalize_forbidden(() if forbidden is None else forbidden, rows, columns)
    _validate_output_keys(row_key, column_key, fields)

    counts = solve_dual_quotas(rows, columns, forbidden_pairs)
    iterator = QuotaRecordIterator(
        counts,
        columns,
        sum(rows.values()),
        generator.random,
        generator,
        row_key,
        column_key,
        fields or {},
    )
    return counts, iterator


def solve_dual_quotas(
    row_quotas: QuotaMapping,
    column_quotas: QuotaMapping,
    forbidden: Any | None = (),
) -> JointCounts:
    rows = _normalize_quotas(row_quotas, "row_quotas")
    columns = _normalize_quotas(column_quotas, "column_quotas")
    forbidden_pairs = _normalize_forbidden(() if forbidden is None else forbidden, rows, columns)

    row_total = sum(rows.values())
    column_total = sum(columns.values())
    if row_total != column_total:
        raise InfeasibleQuotaException(
            f"Quota totals differ: row total is {row_total}, column total is {column_total}.",
            {
                "kind": "total_mismatch",
                "row_total": row_total,
                "column_total": column_total,
                "difference": row_total - column_total,
            },
        )

    row_indices = {row: index for index, row in enumerate(rows)}
    column_indices = {column: index for index, column in enumerate(columns)}
    row_count = len(rows)
    column_count = len(columns)
    source = row_count + column_count
    sink = source + 1
    graph = [{} for _ in range(sink + 1)]

    for row_index, (row, quota) in enumerate(rows.items()):
        graph[source][row_index] = quota
        graph[row_index][source] = 0
        for column_index, column in enumerate(columns):
            column_node = row_count + column_index
            capacity = 0 if (row, column) in forbidden_pairs else row_total
            graph[row_index][column_node] = capacity
            graph[column_node][row_index] = 0

    for column_index, quota in enumerate(columns.values()):
        column_node = row_count + column_index
        graph[column_node][sink] = quota
        graph[sink][column_node] = 0

    flow = _max_flow(graph, source, sink)
    if flow != row_total:
        reachable = _reachable_nodes(graph, source)
        reachable_rows = [row for row, index in row_indices.items() if index in reachable]
        reachable_columns = [
            column
            for column, index in column_indices.items()
            if row_count + index in reachable
        ]
        demand = sum(rows[row] for row in reachable_rows)
        capacity = sum(columns[column] for column in reachable_columns)
        certificate = {
            "kind": "hall_violation",
            "rows": reachable_rows,
            "columns": reachable_columns,
            "demand": demand,
            "capacity": capacity,
            "shortage": demand - capacity,
            "flow": flow,
            "total": row_total,
            "cut": {
                "source_side": {
                    "source": True,
                    "rows": reachable_rows,
                    "columns": reachable_columns,
                },
                "sink_side": {
                    "rows": [row for row in rows if row not in reachable_rows],
                    "columns": [column for column in columns if column not in reachable_columns],
                    "sink": True,
                },
            },
        }
        raise InfeasibleQuotaException(
            f"Quota constraints are infeasible: {demand} units can only use {capacity} allowed capacity.",
            certificate,
        )

    counts: JointCounts = OrderedDict()
    for row_index, row in enumerate(rows):
        counts[row] = OrderedDict()
        for column_index, column in enumerate(columns):
            count = graph[row_count + column_index][row_index]
            if count or (row, column) not in forbidden_pairs:
                counts[row][column] = count
    return counts


def _max_flow(graph: list[dict[int, int]], source: int, sink: int) -> int:
    total = 0
    while True:
        parent = {source: None}
        queue = [source]
        for node in queue:
            if sink in parent:
                break
            for neighbor, capacity in graph[node].items():
                if capacity > 0 and neighbor not in parent:
                    parent[neighbor] = node
                    queue.append(neighbor)

        if sink not in parent:
            return total

        augment = None
        node = sink
        while node != source:
            previous = parent[node]
            capacity = graph[previous][node]
            augment = capacity if augment is None else min(augment, capacity)
            node = previous

        node = sink
        while node != source:
            previous = parent[node]
            graph[previous][node] -= augment
            graph[node][previous] += augment
            node = previous
        total += augment


def _reachable_nodes(graph: list[dict[int, int]], source: int) -> set[int]:
    reachable = {source}
    queue = [source]
    for node in queue:
        for neighbor, capacity in graph[node].items():
            if capacity > 0 and neighbor not in reachable:
                reachable.add(neighbor)
                queue.append(neighbor)
    return reachable


def _normalize_quotas(values: QuotaMapping, name: str) -> OrderedDict[Hashable, int]:
    """Validate and preserve user-provided quota ordering."""
    if not isinstance(values, Mapping):
        raise TypeError(f"{name} must be a mapping.")

    result: OrderedDict[Hashable, int] = OrderedDict()
    for key, value in values.items():
        try:
            hash(key)
        except TypeError:
            raise TypeError(f"{name} keys must be hashable.")
        if type(value) is not int:
            raise TypeError(f"{name} values must be non-negative integers.")
        if value < 0:
            raise ValueError(f"{name} values must be non-negative integers.")
        result[key] = value
    return result


def _normalize_forbidden(
    forbidden: Any,
    rows: QuotaMapping,
    columns: QuotaMapping,
) -> set[tuple[Hashable, Hashable]]:
    """Accept both pair iterables and row-to-forbidden-columns mappings."""
    if isinstance(forbidden, Mapping):
        try:
            pairs: Any = [
                (row, column)
                for row, forbidden_columns in forbidden.items()
                for column in forbidden_columns
            ]
        except TypeError as exc:
            raise TypeError("forbidden mapping values must be iterables of column keys.") from exc
    else:
        pairs = forbidden

    try:
        result = set()
        for pair in pairs:
            if isinstance(pair, (str, bytes)):
                raise ValueError("Forbidden combinations must be (row, column) pairs.")
            try:
                row, column, *extra = tuple(pair)
                if extra:
                    raise ValueError
            except (TypeError, ValueError) as exc:
                raise ValueError("Forbidden combinations must be (row, column) pairs.") from exc
            if row not in rows or column not in columns:
                raise ValueError("Forbidden combinations must use existing quota keys.")
            result.add((row, column))
        return result
    except TypeError as exc:
        raise TypeError("forbidden must be an iterable of pairs or a mapping of rows to columns.") from exc


def _validate_output_keys(row_key: str, column_key: str, fields: ExtraFields | None) -> None:
    if not isinstance(row_key, str) or not isinstance(column_key, str):
        raise TypeError("row_key and column_key must be strings.")
    if row_key == column_key:
        raise ValueError("row_key and column_key must be different.")
    if fields is None:
        return
    if not isinstance(fields, Mapping):
        raise TypeError("fields must be a mapping of field names to formatter names or callables.")
    for name, value in fields.items():
        if not isinstance(name, str):
            raise TypeError("Field names must be strings.")
        if name in (row_key, column_key):
            raise ValueError("Field names cannot duplicate quota value keys.")
        if not isinstance(value, str) and not callable(value):
            raise TypeError("Each field must be a formatter name or zero-argument callable.")
