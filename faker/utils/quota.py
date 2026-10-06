import heapq
import random
from collections import deque
from collections.abc import Collection, Mapping
from dataclasses import dataclass

from typing import Any, Callable, Iterator, List, Optional, Sequence, TypeVar, Union

from ..exceptions import InfeasibleQuotaError

RowKey = TypeVar("RowKey")
ColumnKey = TypeVar("ColumnKey")


@dataclass
class _Edge:
    to: int
    reverse: int
    capacity: int


class _MaxFlow:
    def __init__(self, size: int) -> None:
        self.graph = [[] for _ in range(size)]

    def add_edge(self, source: int, target: int, capacity: int) -> _Edge:
        forward = _Edge(target, len(self.graph[target]), capacity)
        backward = _Edge(source, len(self.graph[source]), 0)
        self.graph[source].append(forward)
        self.graph[target].append(backward)
        return forward

    def max_flow(self, source: int, sink: int, limit: int) -> int:
        total_flow = 0

        while total_flow < limit:
            level = self._bfs(source, sink)
            if level is None:
                break

            next_edge = [0] * len(self.graph)
            while total_flow < limit:
                pushed = self._dfs(source, sink, limit - total_flow, level, next_edge)
                if pushed == 0:
                    break
                total_flow += pushed

        return total_flow

    def reachable(self, source: int) -> tuple[bool, ...]:
        seen = [False] * len(self.graph)
        seen[source] = True
        queue = deque([source])

        while queue:
            node = queue.popleft()
            for edge in self.graph[node]:
                if edge.capacity > 0 and not seen[edge.to]:
                    seen[edge.to] = True
                    queue.append(edge.to)

        return tuple(seen)

    def _bfs(self, source: int, sink: int) -> Optional[List[int]]:
        level = [-1] * len(self.graph)
        level[source] = 0
        queue = deque([source])

        while queue:
            node = queue.popleft()
            for edge in self.graph[node]:
                if edge.capacity > 0 and level[edge.to] == -1:
                    level[edge.to] = level[node] + 1
                    if edge.to == sink:
                        return level
                    queue.append(edge.to)

        return None

    def _dfs(
        self,
        node: int,
        sink: int,
        flow: int,
        level: List[int],
        next_edge: List[int],
    ) -> int:
        if node == sink:
            return flow

        while next_edge[node] < len(self.graph[node]):
            edge = self.graph[node][next_edge[node]]
            if edge.capacity > 0 and level[edge.to] == level[node] + 1:
                pushed = self._dfs(edge.to, sink, min(flow, edge.capacity), level, next_edge)
                if pushed:
                    edge.capacity -= pushed
                    self.graph[edge.to][edge.reverse].capacity += pushed
                    return pushed
            next_edge[node] += 1

        return 0


def categorical_quota_matrix(
    row_quotas: Optional[Mapping[RowKey, int]] = None,
    column_quotas: Optional[Mapping[ColumnKey, int]] = None,
    forbidden_pairs: Union[Collection[Sequence[object]], Mapping[Sequence[object], object]] = (),
) -> dict[RowKey, dict[ColumnKey, int]]:
    row_quotas = {} if row_quotas is None else row_quotas
    column_quotas = {} if column_quotas is None else column_quotas
    rows, columns, allowed = _validated_quotas(row_quotas, column_quotas, forbidden_pairs)
    matrix = {row: {column: 0 for column in columns} for row in rows}
    total = sum(row_quotas.values())
    if total == 0:
        return matrix

    positive_rows = [row for row in rows if row_quotas[row] > 0]
    positive_columns = [column for column in columns if column_quotas[column] > 0]
    positive_row_index = {row: index for index, row in enumerate(positive_rows)}
    positive_column_index = {column: index for index, column in enumerate(positive_columns)}
    row_offset = 1
    column_offset = row_offset + len(positive_rows)
    source = 0
    sink = column_offset + len(positive_columns)
    network = _MaxFlow(sink + 1)

    for row in positive_rows:
        network.add_edge(source, row_offset + positive_row_index[row], row_quotas[row])
    for column in positive_columns:
        network.add_edge(column_offset + positive_column_index[column], sink, column_quotas[column])

    allocation_edges = []
    for full_row_index, row in enumerate(rows):
        if row_quotas[row] == 0:
            continue
        for full_column_index, column in enumerate(columns):
            if column_quotas[column] == 0 or not allowed[full_row_index][full_column_index]:
                continue
            edge = network.add_edge(
                row_offset + positive_row_index[row],
                column_offset + positive_column_index[column],
                total,
            )
            allocation_edges.append((row, column, edge))

    flow = network.max_flow(source, sink, total)
    if flow < total:
        _raise_infeasible(
            network,
            source,
            rows,
            columns,
            positive_rows,
            positive_columns,
            allowed,
            row_quotas,
            column_quotas,
            total - flow,
        )

    for row, column, edge in allocation_edges:
        matrix[row][column] = total - edge.capacity

    return matrix


def iter_categorical_records(
    matrix: Mapping[Any, Mapping[Any, int]],
    random: random.Random,
    row_key: str = "row",
    column_key: str = "column",
    extra_fields: Callable[[Any, Any], Mapping[str, Any]] | None = None,
) -> Iterator[dict[str, Any]]:
    heap = []

    for row, column_counts in matrix.items():
        for column, count in column_counts.items():
            if count > 0:
                key = random.random() ** (1.0 / count)
                heap.append((-key, len(heap), row, column, count, key))

    heapq.heapify(heap)

    while heap:
        _, order, row, column, remaining, key = heap[0]
        if extra_fields is None:
            record = {row_key: row, column_key: column}
        else:
            record = {row_key: row, column_key: column, **extra_fields(row, column)}
        yield record

        remaining -= 1
        if remaining == 0:
            heapq.heappop(heap)
            continue

        next_key = key * random.random() ** (1.0 / remaining)
        heapq.heapreplace(heap, (-next_key, order, row, column, remaining, next_key))


def _validated_quotas(
    row_quotas: Mapping[RowKey, int],
    column_quotas: Mapping[ColumnKey, int],
    forbidden_pairs: Union[Collection[Sequence[object]], Mapping[Sequence[object], object]],
) -> tuple[tuple[RowKey, ...], tuple[ColumnKey, ...], list[list[bool]]]:
    if not isinstance(row_quotas, Mapping) or not isinstance(column_quotas, Mapping):
        raise TypeError("Row and column quotas must be mappings.")

    rows = tuple(row_quotas.keys())
    columns = tuple(column_quotas.keys())
    for quotas, name in ((row_quotas, "Row"), (column_quotas, "Column")):
        for value in quotas.values():
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} quotas must be non-negative integers.")

    row_total = sum(row_quotas.values())
    column_total = sum(column_quotas.values())
    if row_total != column_total:
        raise InfeasibleQuotaError(
            tuple(row for row in rows if row_quotas[row] > 0),
            tuple(column for column in columns if column_quotas[column] > 0),
            row_quotas,
            column_quotas,
            abs(row_total - column_total),
        )

    allowed: list[list[bool]] = [[True] * len(columns) for _ in rows]
    row_index = {row: index for index, row in enumerate(rows)}
    column_index = {column: index for index, column in enumerate(columns)}
    forbidden_items = forbidden_pairs.keys() if isinstance(forbidden_pairs, Mapping) else forbidden_pairs
    for pair in forbidden_items:
        if (
            isinstance(pair, (str, bytes))
            or not isinstance(pair, Sequence)
            or len(pair) != 2
        ):
            raise ValueError("Forbidden pairs must be 2-element sequences of row and column keys.")
        row, column = pair
        if row not in row_index or column not in column_index:
            raise ValueError(f"Forbidden pair {pair!r} contains an unknown quota key.")
        allowed[row_index[row]][column_index[column]] = False

    return rows, columns, allowed


def _raise_infeasible(
    network: _MaxFlow,
    source: int,
    rows: tuple[RowKey, ...],
    columns: tuple[ColumnKey, ...],
    positive_rows: list[RowKey],
    positive_columns: list[ColumnKey],
    allowed: list[list[bool]],
    row_quotas: Mapping[RowKey, int],
    column_quotas: Mapping[ColumnKey, int],
    shortfall: int,
) -> None:
    reachable = network.reachable(source)
    row_offset = 1
    column_offset = row_offset + len(positive_rows)

    witness_rows: tuple[RowKey, ...] = tuple(
        positive_rows[index]
        for index in range(len(positive_rows))
        if reachable[row_offset + index]
    )
    witness_rows_set = set(witness_rows)
    witness_row_indices = {index for index, row in enumerate(rows) if row in witness_rows_set}
    witness_columns: set[ColumnKey] = set(
        positive_columns[index]
        for index in range(len(positive_columns))
        if reachable[column_offset + index]
    )

    for row_index in witness_row_indices:
        for column_index, column in enumerate(columns):
            if allowed[row_index][column_index] and column_quotas[column] == 0:
                witness_columns.add(column)

    witness_column_tuple = tuple(column for column in columns if column in witness_columns)
    supply = sum(row_quotas[row] for row in witness_rows)
    demand = sum(column_quotas[column] for column in witness_column_tuple)

    raise InfeasibleQuotaError(
        witness_rows,
        witness_column_tuple,
        row_quotas,
        column_quotas,
        shortfall,
    )
