from collections import Counter
from collections import OrderedDict

import pytest

from faker import Faker
from faker import InfeasibleQuotaException


def brute_force_feasible(row_quotas, column_quotas, allowed):
    row_names = tuple(row_quotas)
    column_names = tuple(column_quotas)

    def visit(row_index, remaining_columns):
        if row_index == len(row_names):
            return all(remaining == 0 for remaining in remaining_columns)

        quota = row_quotas[row_names[row_index]]
        allowed_columns = tuple(
            index
            for index, column in enumerate(column_names)
            if (row_names[row_index], column) in allowed
        )

        def distribute(quota_left, distribution):
            if len(distribution) == len(allowed_columns):
                if quota_left != 0:
                    return False
                next_columns = list(remaining_columns)
                for column_index, amount in zip(allowed_columns, distribution):
                    next_columns[column_index] -= amount
                if any(value < 0 for value in next_columns):
                    return False
                return visit(row_index + 1, next_columns)

            column_index = allowed_columns[len(distribution)]
            upper = min(quota_left, remaining_columns[column_index])
            return any(
                distribute(quota_left - amount, [*distribution, amount])
                for amount in range(upper + 1)
            )

        return distribute(quota, [])

    return visit(0, list(column_quotas.values()))


def test_dual_quota_records_returns_exact_joint_counts_and_shuffled_records():
    fake = Faker()
    fake.seed_instance(12345)
    quotas = OrderedDict({"east": 30, "south": 20})
    plans = OrderedDict({"basic": 42, "premium": 8})

    counts, records = fake.dual_quota_records(
        quotas,
        plans,
        [("south", "premium")],
        row_key="region",
        column_key="plan",
        fields={"value": "pyint"},
    )

    assert sum(map(sum, (row.values() for row in counts.values()))) == 50
    assert {region: sum(region_counts.values()) for region, region_counts in counts.items()} == quotas
    assert {
        plan: sum(counts[region].get(plan, 0) for region in counts)
        for plan in plans
    } == plans
    assert "premium" not in counts["south"]

    output = list(records)
    assert len(output) == 50
    assert Counter((row["region"], row["plan"]) for row in output) == Counter(
        {(region, plan): count for region, region_counts in counts.items() for plan, count in region_counts.items()}
    )
    assert all({"region", "plan", "value"} <= set(row) for row in output)
    assert output != sorted(output, key=lambda row: (row["region"], row["plan"]))


def test_dual_quota_records_are_reproducible_with_instance_random():
    def generate():
        fake = Faker()
        fake.seed_instance(7)
        _, records = fake.dual_quota_records(
            OrderedDict([("east", 5), ("south", 3)]),
            OrderedDict([("basic", 6), ("premium", 2)]),
            {("south", "premium")},
            fields={"number": "pyint"},
        )
        return list(records)

    assert generate() == generate()


def test_infeasible_forbidden_combination_has_reviewable_gap_and_cut():
    fake = Faker()
    with pytest.raises(InfeasibleQuotaException) as exc_info:
        fake.dual_quota_records(
            OrderedDict([("east", 2), ("south", 3)]),
            OrderedDict([("basic", 1), ("premium", 4)]),
            [("south", "premium")],
        )

    certificate = exc_info.value.certificate
    assert certificate["kind"] == "hall_violation"
    assert certificate["rows"] == ["south"]
    assert certificate["columns"] == ["basic"]
    assert certificate["demand"] == 3
    assert certificate["capacity"] == 1
    assert certificate["shortage"] == 2
    assert certificate["cut"]["source_side"]["rows"] == ["south"]
    assert certificate["cut"]["source_side"]["columns"] == ["basic"]
    assert certificate["cut"]["sink_side"]["columns"] == ["premium"]


def test_total_mismatch_reports_supply_demand_gap():
    with pytest.raises(InfeasibleQuotaException) as exc_info:
        Faker().dual_quota_records({"east": 2}, {"basic": 3})

    assert exc_info.value.certificate == {
        "kind": "total_mismatch",
        "row_total": 2,
        "column_total": 3,
        "difference": -1,
    }


def test_small_quota_feasibility_matches_brute_force():
    from faker.quota import solve_dual_quotas

    row_quotas = {"r0": 1, "r1": 2}
    column_quotas = {"c0": 1, "c1": 2}
    all_pairs = [("r0", "c0"), ("r0", "c1"), ("r1", "c0"), ("r1", "c1")]

    for mask in range(1 << len(all_pairs)):
        allowed = {pair for index, pair in enumerate(all_pairs) if mask & (1 << index)}
        forbidden = [pair for pair in all_pairs if pair not in allowed]
        expected = brute_force_feasible(row_quotas, column_quotas, allowed)
        if expected:
            counts = solve_dual_quotas(row_quotas, column_quotas, forbidden)
            assert sum(counts["r0"].values()) == 1
            assert sum(counts["r1"].values()) == 2
            assert counts["r0"].get("c0", 0) + counts["r1"].get("c0", 0) == 1
            assert counts["r0"].get("c1", 0) + counts["r1"].get("c1", 0) == 2
        else:
            with pytest.raises(InfeasibleQuotaException):
                solve_dual_quotas(row_quotas, column_quotas, forbidden)


def test_large_result_is_consumed_as_iterator_without_count_matrix_growth():
    total = 1_000_000
    fake = Faker()
    fake.seed_instance(99)
    counts, records = fake.dual_quota_records(
        OrderedDict([("east", total // 2), ("south", total // 2)]),
        OrderedDict([("basic", total // 2), ("premium", total // 2)]),
        [("south", "premium")],
    )

    assert len(records._remaining) == 2
    assert all(len(row) == 2 for row in records._remaining)

    seen_regions = Counter()
    seen_plans = Counter()
    consumed = 0
    for record in records:
        seen_regions[record["row"]] += 1
        seen_plans[record["column"]] += 1
        consumed += 1

    assert consumed == total
    assert seen_regions == Counter({"east": total // 2, "south": total // 2})
    assert seen_plans == Counter({"basic": total // 2, "premium": total // 2})


def test_invalid_quotas_are_rejected():
    with pytest.raises(ValueError):
        Faker().dual_quota_records({"east": -1}, {"basic": -1})
    with pytest.raises(TypeError):
        Faker().dual_quota_records({"east": 1.0}, {"basic": 1})
    with pytest.raises(ValueError):
        Faker().dual_quota_records({"east": 1}, {"basic": 1}, [("east", "unknown")])
    with pytest.raises(ValueError):
        Faker().dual_quota_records({"east": 1}, {"basic": 1}, ["east"])


def test_mapping_forbidden_and_callable_fields_are_supported():
    fake = Faker()
    fake.seed_instance(4)
    counts, records = fake.dual_quota_records(
        OrderedDict([("east", 2), ("south", 1)]),
        OrderedDict([("basic", 1), ("premium", 2)]),
        {"south": ["premium"]},
        fields={"constant": lambda: "ok"},
    )

    assert "premium" not in counts["south"]
    assert all(record["constant"] == "ok" for record in records)


def test_quota_record_aliases_share_the_same_entry_point():
    fake = Faker()
    counts, records = fake.joint_quota_records({"east": 1}, {"basic": 1})

    assert counts == OrderedDict([("east", OrderedDict([("basic", 1)]))])
    assert list(records) == [{"row": "east", "column": "basic"}]
