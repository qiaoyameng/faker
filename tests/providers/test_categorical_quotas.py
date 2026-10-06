from collections import Counter
from itertools import product

import pytest

from faker import Faker
from faker.exceptions import InfeasibleQuotaError


def _brute_force_feasible(row_quotas, column_quotas, forbidden_pairs):
    rows = tuple(row_quotas)
    columns = tuple(column_quotas)
    forbidden = set(forbidden_pairs)

    def fill(row_index, remaining_columns):
        if row_index == len(rows):
            return all(value == 0 for value in remaining_columns)

        row = rows[row_index]
        quota = row_quotas[row]

        def choose(column_index, remaining_quota, columns_left):
            if column_index == len(columns):
                return remaining_quota == 0 and fill(row_index + 1, columns_left)

            column = columns[column_index]
            maximum = min(remaining_quota, columns_left[column_index])
            minimum = 0
            later_capacity = sum(columns_left[index] for index in range(column_index + 1, len(columns)))
            if later_capacity < remaining_quota:
                minimum = remaining_quota - later_capacity

            if (row, column) in forbidden:
                maximum = 0

            for count in range(minimum, maximum + 1):
                next_columns = list(columns_left)
                next_columns[column_index] -= count
                if choose(column_index + 1, remaining_quota - count, next_columns):
                    return True
            return False

        return choose(0, quota, list(remaining_columns))

    return fill(0, [column_quotas[column] for column in columns])


def test_small_quota_cases_match_exhaustive_check():
    for row_values in product(range(3), repeat=2):
        for column_values in product(range(3), repeat=2):
            if sum(row_values) != sum(column_values):
                continue

            rows = {f"row-{index}": value for index, value in enumerate(row_values)}
            columns = {f"column-{index}": value for index, value in enumerate(column_values)}

            for forbidden_mask in range(1 << 4):
                forbidden = {
                    (row_name, column_name)
                    for index, (row_name, column_name) in enumerate(product(rows, columns))
                    if forbidden_mask & (1 << index)
                }

                if _brute_force_feasible(rows, columns, forbidden):
                    matrix = Faker().categorical_quota_matrix(rows, columns, forbidden)
                    assert {row: sum(counts.values()) for row, counts in matrix.items()} == rows
                    assert {
                        column: sum(matrix[row][column] for row in rows) for column in columns
                    } == columns
                    assert all(matrix[row][column] == 0 for row, column in forbidden)
                else:
                    with pytest.raises(InfeasibleQuotaError):
                        Faker().categorical_quota_matrix(rows, columns, forbidden)


def test_records_preserve_exact_quotas_and_forbidden_pairs():
    fake = Faker()
    fake.seed_instance(2026)
    regions = {"east": 7, "south": 5, "west": 3}
    plans = {"basic": 8, "standard": 4, "premium": 3}
    forbidden = {("west", "premium"), ("south", "standard")}

    records = list(
        fake.random_categorical_records(
            regions,
            plans,
            forbidden,
            row_key="region",
            column_key="plan",
            extra_fields=lambda region, plan: {"source": f"{region}-{plan}"},
        )
    )

    assert dict(Counter(record["region"] for record in records)) == regions
    assert dict(Counter(record["plan"] for record in records)) == plans
    assert not any((record["region"], record["plan"]) in forbidden for record in records)
    assert all(record["source"] == f"{record['region']}-{record['plan']}" for record in records)


def test_record_order_is_seedable_but_matrix_does_not_consume_random_state():
    row_quotas = {"a": 2, "b": 2}
    column_quotas = {"x": 2, "y": 2}

    first = Faker()
    first.seed_instance(42)
    first_matrix = first.categorical_quota_matrix(row_quotas, column_quotas)
    first_order = list(first.random_categorical_records(row_quotas, column_quotas))

    second = Faker()
    second.seed_instance(42)
    second_matrix = second.categorical_quota_matrix(row_quotas, column_quotas)
    second_order = list(second.random_categorical_records(row_quotas, column_quotas))

    third = Faker()
    third.seed_instance(43)
    third_matrix = third.categorical_quota_matrix(row_quotas, column_quotas)
    third_order = list(third.random_categorical_records(row_quotas, column_quotas))

    assert first_matrix == second_matrix == third_matrix
    assert first_order == second_order
    assert first_order != third_order


def test_infeasible_quota_error_reports_verifiable_shortfall():
    fake = Faker()
    row_quotas = {"east": 2, "south": 2}
    column_quotas = {"basic": 1, "premium": 3}

    with pytest.raises(InfeasibleQuotaError) as exc_info:
        fake.categorical_quota_matrix(
            row_quotas,
            column_quotas,
            {("east", "basic"), ("south", "basic")},
        )

    error = exc_info.value
    supply = sum(row_quotas[row] for row in error.row_subset)
    demand = sum(column_quotas[column] for column in error.column_subset)
    assert supply - demand == error.shortfall > 0


def test_infeasible_records_fail_before_iterator_is_returned():
    fake = Faker()
    with pytest.raises(InfeasibleQuotaError):
        fake.random_categorical_records(
            {"a": 2},
            {"x": 2},
            {("a", "x")},
        )


@pytest.mark.parametrize(
    ("rows", "columns"),
    (
        ({"a": -1}, {"x": -1}),
        ({"a": 1}, {"x": 1.0}),
    ),
)
def test_invalid_quota_input(rows, columns):
    with pytest.raises(ValueError):
        Faker().categorical_quota_matrix(rows, columns)


def test_unequal_quota_totals_report_business_shortfall():
    rows = {"a": 1, "b": 1}
    columns = {"x": 3}

    with pytest.raises(InfeasibleQuotaError) as exc_info:
        Faker().categorical_quota_matrix(rows, columns)

    error = exc_info.value
    assert error.row_subset == ("a", "b")
    assert error.column_subset == ("x",)
    assert error.shortfall == 1


def test_unknown_forbidden_pair_is_rejected():
    with pytest.raises(ValueError, match="unknown quota key"):
        Faker().categorical_quota_matrix({"a": 1}, {"x": 1}, {("a", "z")})


def test_forbidden_pairs_accept_sequences_but_not_strings():
    matrix = Faker().categorical_quota_matrix(
        {"a": 1, "b": 1},
        {"x": 1, "y": 1},
        [["a", "x"], ["b", "y"]],
    )

    assert matrix["a"]["x"] == 0
    assert matrix["b"]["y"] == 0

    with pytest.raises(ValueError, match="2-element sequences"):
        Faker().categorical_quota_matrix({"a": 1}, {"x": 1}, ["ax"])


def test_forbidden_pairs_accept_mapping_of_pair_to_flag():
    matrix = Faker().categorical_quota_matrix(
        {"a": 1, "b": 1},
        {"x": 1, "y": 1},
        {("a", "x"): True},
    )

    assert matrix["a"]["x"] == 0
    assert matrix["a"]["y"] == 1


def test_zero_quotas_keep_all_categories_and_forbidden_pairs():
    rows = {"a": 0, "b": 0}
    columns = {"x": 0, "y": 0}
    matrix = Faker().categorical_quota_matrix(rows, columns, {("a", "x")})

    assert matrix == {"a": {"x": 0, "y": 0}, "b": {"x": 0, "y": 0}}
    assert list(Faker().random_categorical_records(rows, columns)) == []


def test_million_records_are_consumed_from_iterator():
    fake = Faker()
    fake.seed_instance(7)
    rows = {f"row-{index}": 100_000 for index in range(10)}
    columns = {f"column-{index}": 100_000 for index in range(10)}

    records = fake.random_categorical_records(rows, columns)
    first_record = next(records)
    assert len(records.gi_frame.f_locals["heap"]) <= len(rows) * len(columns)

    row_counts = Counter()
    column_counts = Counter()
    row_counts[first_record["row"]] += 1
    column_counts[first_record["column"]] += 1

    for record in records:
        row_counts[record["row"]] += 1
        column_counts[record["column"]] += 1

    assert dict(row_counts) == rows
    assert dict(column_counts) == columns
