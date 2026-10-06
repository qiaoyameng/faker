"""Tests for the relational dataset generator (``faker.relations``).

The relationship checks in these tests intentionally reimplement referential
integrity as independent graph traversals over the generated rows. They do not
call any private validation helper from the generator, so a broken generator
cannot be proven correct by its own checks.
"""

import pytest

from faker import Faker
from faker.relations import (
    Entity,
    EntitySpec,
    Reference,
    RelationDefinitionError,
    UnsatisfiableRelationError,
    generate_dataset,
)


def _key_index(rows, pk):
    """Build a primary-key -> row index map directly from the output rows."""

    index = {}
    for position, row in enumerate(rows):
        value = row[pk]
        assert value not in index, f"duplicate primary key {value!r}"
        index[value] = position
    return index


def _assert_integrity(dataset, sources, targets, pk_field, fk_field, *, unique, nullable):
    """Independently traverse source -> target edges and verify every rule."""

    source_rows = dataset[sources]
    target_rows = dataset[targets]
    target_index = _key_index(target_rows, pk_field)

    seen_targets = set()
    null_count = 0
    edges = []
    for position, row in enumerate(source_rows):
        assert fk_field in row, f"missing foreign-key column on row {position}"
        value = row[fk_field]
        if value is None:
            null_count += 1
            assert nullable, f"required reference {sources}.{fk_field} has a null on row {position}"
            continue
        assert value in target_index, (
            f"dangling reference {sources}.{fk_field}={value!r} on row {position}"
        )
        if unique:
            assert value not in seen_targets, (
                f"unique reference {sources}.{fk_field} reuses target {value!r}"
            )
            seen_targets.add(value)
        edges.append((position, target_index[value]))

    return {"edges": edges, "nulls": null_count, "used": seen_targets}


@pytest.fixture
def faker():
    fake = Faker()
    Faker.seed(0)
    return fake


class TestCircularDepartmentEmployee:
    """Sample 3: department.manager_id <-> employee.department_id."""

    def _build(self, department_count, employee_count):
        departments = EntitySpec(
            Entity("department", count=department_count, fields={"name": "company"}),
            references=[Reference("manager_id", "employee", unique=True)],
        )
        employees = EntitySpec(
            Entity("employee", count=employee_count, fields={"name": "name"}),
            references=[Reference("department_id", "department")],
        )
        return departments, employees

    def test_circular_required_references_are_both_satisfied(self, faker):
        specs = self._build(department_count=3, employee_count=9)

        dataset = generate_dataset(specs, faker, seed=11)

        _assert_integrity(
            dataset,
            "department",
            "employee",
            "id",
            "manager_id",
            unique=True,
            nullable=False,
        )
        _assert_integrity(
            dataset,
            "employee",
            "department",
            "id",
            "department_id",
            unique=False,
            nullable=False,
        )
        assert len({row["manager_id"] for row in dataset["department"]}) == 3
        assert {row["department_id"] for row in dataset["employee"]} <= {0, 1, 2}

    def test_too_few_employees_for_unique_managers_is_unsatisfiable(self, faker):
        specs = self._build(department_count=5, employee_count=2)

        with pytest.raises(UnsatisfiableRelationError) as excinfo:
            generate_dataset(specs, faker)
        assert excinfo.value.reference == "department.manager_id"
        assert excinfo.value.required_size == 5

class TestDefinitionErrors:
    def test_unknown_reference_target(self, faker):
        spec = EntitySpec(Entity("x", count=1), references=[Reference("y_id", "missing")])
        with pytest.raises(RelationDefinitionError, match="unknown entity 'missing'"):
            generate_dataset([spec], faker)

    def test_reference_field_clashes_with_regular_field(self, faker):
        spec = EntitySpec(
            Entity("x", count=1, fields={"y_id": "name"}),
            references=[Reference("y_id", "x")],
        )
        with pytest.raises(RelationDefinitionError, match="clashes"):
            generate_dataset([spec], faker)

    def test_negative_count_rejected(self, faker):
        with pytest.raises(RelationDefinitionError, match="non-negative integer"):
            generate_dataset([Entity("x", count=-1)], faker)

    def test_invalid_null_fraction_rejected(self, faker):
        spec = EntitySpec(Entity("x", count=1), references=[Reference("y", "x", null_fraction=1.5)])
        with pytest.raises(RelationDefinitionError, match="null_fraction"):
            generate_dataset([spec], faker)

    def test_no_entities_rejected(self, faker):
        with pytest.raises(RelationDefinitionError, match="At least one entity"):
            generate_dataset([], faker)

    def test_invalid_provider_rejected(self, faker):
        with pytest.raises(RelationDefinitionError, match="provider must be"):
            generate_dataset([Entity("x", count=1, fields={"v": 123})], faker)

    def test_duplicate_primary_key_values_rejected(self, faker):
        with pytest.raises(RelationDefinitionError, match="duplicate values"):
            generate_dataset(
                [Entity("x", count=2, fields={"code": lambda: "same"}, primary_key="code")],
                faker,
            )

class TestEmployeeSupervisorSelfReference:
    """Sample 1: employees pointing at other employees (nullable many-to-one)."""

    def test_self_reference_rows_point_to_existing_employees(self, faker):
        employee = EntitySpec(
            Entity("employee", count=20, fields={"name": "name", "badge": "pyint"}),
            references=[
                Reference("supervisor_id", "employee", nullable=True, null_fraction=0.25)
            ],
        )

        dataset = generate_dataset([employee], faker, seed=7)

        rows = dataset["employee"]
        assert len(rows) == 20
        stats = _assert_integrity(
            dataset,
            "employee",
            "employee",
            "id",
            "supervisor_id",
            unique=False,
            nullable=True,
        )
        assert stats["nulls"] > 0
        assert stats["edges"], "expected at least one supervisor assignment"

    def test_required_self_reference_never_null(self, faker):
        employee = EntitySpec(
            Entity("employee", count=12, fields={"name": "name"}),
            references=[Reference("manager_id", "employee")],
        )

        dataset = generate_dataset([employee], faker, seed=1)

        stats = _assert_integrity(
            dataset, "employee", "employee", "id", "manager_id", unique=False, nullable=False
        )
        assert stats["nulls"] == 0
        assert len(stats["edges"]) == 12

    def test_required_self_reference_into_empty_table_is_unsatisfiable(self, faker):
        employee = EntitySpec(
            Entity("employee", count=0),
            references=[Reference("manager_id", "employee")],
        )

        with pytest.raises(UnsatisfiableRelationError) as excinfo:
            generate_dataset([employee], faker)
        assert excinfo.value.reference == "employee.manager_id"
        assert excinfo.value.required_size == 1

class TestNullableAndProviderBehavior:
    def test_nullable_unique_forces_extra_rows_null_when_targets_run_short(self, faker):
        cards = EntitySpec(
            Entity("card", count=6, fields={"label": "word"}),
            references=[Reference("holder_id", "person", unique=True, nullable=True, null_fraction=0.0)],
        )
        people = Entity("person", count=4, fields={"name": "name"})

        dataset = generate_dataset([cards, people], faker, seed=5)

        stats = _assert_integrity(
            dataset, "card", "person", "id", "holder_id", unique=True, nullable=True
        )
        assert len(stats["used"]) == 4
        assert stats["nulls"] == 2

    def test_nullable_reference_into_empty_table_leaves_all_null(self, faker):
        cards = EntitySpec(
            Entity("card", count=3, fields={"label": "word"}),
            references=[Reference("holder_id", "person", unique=True, nullable=True)],
        )
        people = Entity("person", count=0)

        dataset = generate_dataset([cards, people], faker)

        assert [row["holder_id"] for row in dataset["card"]] == [None, None, None]

    def test_null_fraction_zero_on_nullable_unique_still_fills_when_possible(self, faker):
        cards = EntitySpec(
            Entity("card", count=3, fields={"label": "word"}),
            references=[Reference("holder_id", "person", unique=True, nullable=True, null_fraction=0.0)],
        )
        people = Entity("person", count=3, fields={"name": "name"})

        dataset = generate_dataset([cards, people], faker, seed=2)

        stats = _assert_integrity(
            dataset, "card", "person", "id", "holder_id", unique=True, nullable=True
        )
        assert stats["nulls"] == 0
        assert len(stats["used"]) == 3

    def test_callable_providers_and_custom_primary_keys(self, faker):
        codes = iter(range(100))
        widgets = Entity(
            "widget",
            count=4,
            fields={"code": codes.__next__, "label": lambda: "fixed"},
            primary_key="code",
        )

        dataset = generate_dataset([widgets], faker)

        rows = dataset["widget"]
        assert [row["code"] for row in rows] == [0, 1, 2, 3]
        assert all(row["label"] == "fixed" for row in rows)

    def test_same_seed_produces_same_assignments(self, faker):
        employee = EntitySpec(
            Entity("employee", count=15, fields={"name": "name"}),
            references=[Reference("supervisor_id", "employee", nullable=True, null_fraction=0.5)],
        )

        first = generate_dataset([employee], Faker(), seed=42)
        second = generate_dataset([employee], Faker(), seed=42)

        first_values = [row["supervisor_id"] for row in first["employee"]]
        second_values = [row["supervisor_id"] for row in second["employee"]]
        assert first_values == second_values

class TestUserUniqueProfile:
    """Sample 2: user <-> profile, a pair of one-to-one references."""

    def _build(self, user_count, profile_count):
        users = EntitySpec(
            Entity("user", count=user_count, fields={"email": "email", "name": "name"}),
            references=[Reference("profile_id", "profile", unique=True)],
        )
        profiles = EntitySpec(
            Entity("profile", count=profile_count, fields={"bio": "sentence"}),
            references=[Reference("user_id", "user", unique=True)],
        )
        return users, profiles

    def test_bijection_between_users_and_profiles(self, faker):
        specs = self._build(user_count=10, profile_count=10)

        dataset = generate_dataset(specs, faker, seed=3)

        forward = _assert_integrity(
            dataset, "user", "profile", "id", "profile_id", unique=True, nullable=False
        )
        _assert_integrity(
            dataset, "profile", "user", "id", "user_id", unique=True, nullable=False
        )

        user_to_profile = {row["id"]: row["profile_id"] for row in dataset["user"]}
        profile_to_user = {row["id"]: row["user_id"] for row in dataset["profile"]}
        # Both directions are independently valid one-to-one references; the
        # generator does not declare references as paired inverses.
        assert set(user_to_profile.values()) == set(range(10))
        assert set(profile_to_user.values()) == set(range(10))
        assert len(forward["edges"]) == 10

    def test_more_users_than_profiles_is_unsatisfiable(self, faker):
        specs = self._build(user_count=11, profile_count=10)

        with pytest.raises(UnsatisfiableRelationError) as excinfo:
            generate_dataset(specs, faker)
        assert excinfo.value.reference == "user.profile_id"
        assert excinfo.value.required_size == 11

    def test_more_profiles_than_users_is_unsatisfiable(self, faker):
        specs = self._build(user_count=10, profile_count=12)

        with pytest.raises(UnsatisfiableRelationError) as excinfo:
            generate_dataset(specs, faker)
        assert excinfo.value.reference == "profile.user_id"
        assert excinfo.value.required_size == 12
