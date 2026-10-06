import pytest

from faker import Faker
from faker.contrib.relational import (
    Entity,
    Ref,
    RelationalDatasetGenerator,
    RelationalSchemaError,
    UnsatisfiableRelationError,
    generate_dataset,
)
from faker.providers import BaseProvider


def index_by(rows, key):
    """Build a lookup of rows by ``key``, asserting the key is unique."""
    indexed = {}
    for row in rows:
        assert row[key] not in indexed
        indexed[row[key]] = row
    return indexed


def assert_references_hold(dataset, source, fk, target, target_key, required, unique=False):
    """Independently walk a dataset and check one foreign-key relation."""
    target_keys = {row[target_key] for row in dataset[target]}
    seen = set()
    for row in dataset[source]:
        value = row[fk]
        if value is None:
            assert not required, f"{source}.{fk} unexpectedly None"
            continue
        assert value in target_keys, f"{source}.{fk}={value!r} missing from {target}.{target_key}"
        if unique:
            assert value not in seen, f"{source}.{fk}={value!r} used twice"
            seen.add(value)
    return seen


class TestEmployeeSupervisorSample:
    """Self-referencing sample: employees and their (optional) supervisors."""

    def build(self, seed=20261006):
        employee = Entity(
            name="employee",
            count=12,
            pk="id",
            fields={"full_name": "name", "email": "email"},
            refs=(
                Ref(field="supervisor_id", target="employee", required=False, allow_self=False),
            ),
        )
        return generate_dataset([employee], seed=seed)

    def test_shape_and_fields(self):
        dataset = self.build()
        assert set(dataset) == {"employee"}
        assert len(dataset["employee"]) == 12
        for row in dataset["employee"]:
            assert set(row) == {"id", "full_name", "email", "supervisor_id"}
            assert isinstance(row["full_name"], str) and row["full_name"]
            assert isinstance(row["email"], str) and row["email"]

    def test_supervisor_chain_by_independent_traversal(self):
        dataset = self.build()
        employees = index_by(dataset["employee"], "id")
        assert sorted(employees) == list(range(1, 13))
        for row in dataset["employee"]:
            supervisor_id = row["supervisor_id"]
            assert supervisor_id != row["id"], "employee must not supervise themselves"
            if supervisor_id is not None:
                assert supervisor_id in employees
        # Walk every chain: each step stays inside the dataset and never
        # immediately loops back onto the same employee.
        for row in dataset["employee"]:
            current = row
            for _ in range(len(employees)):
                next_id = current["supervisor_id"]
                if next_id is None:
                    break
                current = employees[next_id]
                assert current["id"] != row["id"] or current is not row

    def test_deterministic_for_same_seed(self):
        assert self.build(seed=7) == self.build(seed=7)


class TestUserProfileSample:
    """One-to-one sample: users and their unique, mandatory profiles."""

    def build(self, seed=99):
        user = Entity(
            name="user",
            count=8,
            pk="id",
            fields={"username": "user_name"},
        )
        profile = Entity(
            name="profile",
            count=8,
            pk="id",
            fields={"bio": "sentence"},
            refs=(Ref(field="user_id", target="user", required=True, unique=True),),
        )
        return generate_dataset([user, profile], seed=seed)

    def test_profiles_form_bijection_with_users(self):
        dataset = self.build()
        users = index_by(dataset["user"], "id")
        index_by(dataset["profile"], "id")
        used = assert_references_hold(dataset, "profile", "user_id", "user", "id", required=True, unique=True)
        # Same row counts + every reference valid and distinct => bijection.
        assert used == set(users)

    def test_too_few_users_is_unsatisfiable(self):
        user = Entity(name="user", count=3, pk="id")
        profile = Entity(
            name="profile",
            count=5,
            pk="id",
            refs=(Ref(field="user_id", target="user", required=True, unique=True),),
        )
        with pytest.raises(UnsatisfiableRelationError) as excinfo:
            generate_dataset([user, profile], seed=1)
        assert "profile.user_id -> user.id" in str(excinfo.value)


class TestCircularReferenceSample:
    """Two entities referencing each other: projects and their tasks."""

    def build(self, seed=4242):
        project = Entity(
            name="project",
            count=5,
            pk="id",
            fields={"title": "catch_phrase"},
            refs=(Ref(field="lead_task_id", target="task", required=False),),
        )
        task = Entity(
            name="task",
            count=9,
            pk="id",
            fields={"label": "word"},
            refs=(Ref(field="project_id", target="project", required=True),),
        )
        return generate_dataset([project, task], seed=seed)

    def test_both_directions_hold(self):
        dataset = self.build()
        projects = index_by(dataset["project"], "id")
        tasks = index_by(dataset["task"], "id")
        assert_references_hold(dataset, "task", "project_id", "project", "id", required=True)
        assert_references_hold(dataset, "project", "lead_task_id", "task", "id", required=False)
        # Cross the cycle: task -> project -> lead task must stay navigable.
        for task in dataset["task"]:
            project = projects[task["project_id"]]
            lead_task_id = project["lead_task_id"]
            if lead_task_id is not None:
                assert lead_task_id in tasks


class TestUnsatisfiableRelations:
    def test_required_reference_to_empty_entity(self):
        empty = Entity(name="empty", count=0, pk="id")
        dependent = Entity(
            name="dependent",
            count=2,
            pk="id",
            refs=(Ref(field="empty_id", target="empty", required=True),),
        )
        with pytest.raises(UnsatisfiableRelationError) as excinfo:
            generate_dataset([empty, dependent], seed=1)
        assert "dependent.empty_id -> empty.id" in str(excinfo.value)

    def test_nullable_reference_to_empty_entity_is_fine(self):
        empty = Entity(name="empty", count=0, pk="id")
        dependent = Entity(
            name="dependent",
            count=3,
            pk="id",
            refs=(Ref(field="empty_id", target="empty", required=False),),
        )
        dataset = generate_dataset([empty, dependent], seed=1)
        assert dataset["empty"] == []
        assert all(row["empty_id"] is None for row in dataset["dependent"])

    def test_single_row_cannot_avoid_self_reference(self):
        node = Entity(
            name="node",
            count=1,
            pk="id",
            refs=(Ref(field="parent_id", target="node", required=True, allow_self=False),),
        )
        with pytest.raises(UnsatisfiableRelationError) as excinfo:
            generate_dataset([node], seed=1)
        assert "node.parent_id -> node.id" in str(excinfo.value)

    def test_no_partial_dataset_on_failure(self):
        ok = Entity(name="ok", count=4, pk="id", fields={"word": "word"})
        broken = Entity(
            name="broken",
            count=10,
            pk="id",
            refs=(Ref(field="ok_id", target="ok", required=True, unique=True),),
        )
        generator = RelationalDatasetGenerator(seed=3)
        with pytest.raises(UnsatisfiableRelationError):
            generator.generate([ok, broken])
        # A second, satisfiable call on the same generator still works and
        # returns a whole dataset, proving nothing half-generated lingers.
        dataset = generator.generate([ok, Entity(name="fine", count=2, pk="id")])
        assert set(dataset) == {"ok", "fine"}
        assert len(dataset["ok"]) == 4
        assert len(dataset["fine"]) == 2


class TestSchemaValidation:
    def test_unknown_provider_rejected(self):
        entity = Entity(name="thing", count=1, pk="id", fields={"x": "not_a_provider"})
        with pytest.raises(RelationalSchemaError, match="not_a_provider"):
            generate_dataset([entity], seed=1)

    def test_non_string_provider_rejected(self):
        entity = Entity(name="thing", count=1, pk="id", fields={"x": lambda: "code"})
        with pytest.raises(RelationalSchemaError):
            generate_dataset([entity], seed=1)

    def test_private_provider_name_rejected(self):
        entity = Entity(name="thing", count=1, pk="id", fields={"x": "_private"})
        with pytest.raises(RelationalSchemaError):
            generate_dataset([entity], seed=1)

    def test_reference_to_unknown_entity_rejected(self):
        entity = Entity(
            name="thing",
            count=1,
            pk="id",
            refs=(Ref(field="ghost_id", target="ghost"),),
        )
        with pytest.raises(RelationalSchemaError, match="ghost"):
            generate_dataset([entity], seed=1)

    def test_duplicate_entity_names_rejected(self):
        entity = Entity(name="thing", count=1, pk="id")
        with pytest.raises(RelationalSchemaError, match="Duplicate"):
            generate_dataset([entity, entity], seed=1)

    def test_reference_field_collision_rejected(self):
        entity = Entity(
            name="thing",
            count=1,
            pk="id",
            fields={"label": "word"},
            refs=(Ref(field="label", target="thing"),),
        )
        with pytest.raises(RelationalSchemaError, match="collides"):
            generate_dataset([entity], seed=1)

    def test_duplicate_primary_key_values_rejected(self):
        class ConstantProvider(BaseProvider):
            def constant_id(self):
                return 1

        fake = Faker()
        fake.add_provider(ConstantProvider)
        entity = Entity(name="thing", count=2, pk="id", fields={"id": "constant_id"})
        with pytest.raises(RelationalSchemaError, match="duplicate"):
            generate_dataset([entity], faker=fake, seed=1)

    def test_custom_provider_used_for_plain_fields(self):
        class ConstantProvider(BaseProvider):
            def constant_value(self):
                return "constant"

        fake = Faker()
        fake.add_provider(ConstantProvider)
        entity = Entity(name="thing", count=3, pk="id", fields={"marker": "constant_value"})
        dataset = generate_dataset([entity], faker=fake, seed=1)
        assert [row["marker"] for row in dataset["thing"]] == ["constant"] * 3
        assert [row["id"] for row in dataset["thing"]] == [1, 2, 3]
