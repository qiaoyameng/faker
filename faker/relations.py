"""Small relational dataset generator.

This module turns declarative entity and reference descriptions into a whole
dataset that satisfies primary-key and foreign-key constraints. Regular column
values are produced by existing Faker providers (or any zero-argument
callable); the generator only owns primary keys and references.

References are described with structured ``Reference`` objects, never with
arbitrary code strings, and generation is entirely in-memory: no database
connection is involved.

Example:
    >>> from faker import Faker
    >>> from faker.relations import Entity, EntitySpec, Reference, generate_dataset
    >>> fake = Faker()
    >>> Faker.seed(0)
    >>> departments = EntitySpec(
    ...     Entity("department", count=2, fields={"name": "company"}),
    ...     references=[Reference("head_id", "employee", unique=True)],
    ... )
    >>> employees = EntitySpec(
    ...     Entity(
    ...         "employee",
    ...         count=4,
    ...         fields={"name": "name"},
    ...     ),
    ...     references=[Reference("department_id", "department")],
    ... )
"""

import random as random_module

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Sequence, Union

from .typing import SeedType

Provider = Union[str, Callable[[], Any]]


class RelationError(Exception):
    """Base class for errors raised by the relational dataset generator."""


class RelationDefinitionError(RelationError):
    """Raised when an entity or reference description is malformed."""


class UnsatisfiableRelationError(RelationError):
    """Raised when the declared cardinalities cannot be satisfied.

    ``reference`` identifies the offending reference as ``"Entity.field"`` and
    ``required_size`` is the minimum number of target rows that would make the
    relation satisfiable.
    """

    def __init__(self, message: str, reference: str, required_size: int) -> None:
        self.reference = reference
        self.required_size = required_size
        super().__init__(message)


@dataclass
class Reference:
    """Description of a many-to-one or one-to-one foreign-key column.

    The column is added to the source entity and populated with primary-key
    values of the target entity.

    :param field: column name on the source entity.
    :param target: name of the target entity; may equal the source name for a
        self reference.
    :param unique: when ``True`` each target primary key is used at most once,
        modelling a one-to-one reference; when ``False`` targets may be shared,
        modelling a many-to-one reference.
    :param nullable: when ``True`` the column may be ``None``; when ``False``
        every row must reference an existing target.
    :param null_fraction: probability in ``[0, 1]`` that a nullable row is left
        as ``None``. Ignored for required references. When a unique nullable
        reference runs short of distinct targets, extra rows are forced to
        ``None`` so the relation always remains satisfiable.
    """

    field: str
    target: str
    unique: bool = False
    nullable: bool = False
    null_fraction: float = 0.0

    @property
    def cardinality(self) -> str:
        """Human-readable cardinality of the reference."""

        return "one_to_one" if self.unique else "many_to_one"


@dataclass
class Entity:
    """Description of an entity table.

    :param name: unique entity name, referenced by ``Reference.target``.
    :param count: number of rows to generate; must be a non-negative integer.
    :param fields: mapping of column name to a Faker formatter name (str) or a
        zero-argument callable producing the column value.
    :param primary_key: name of the column holding the primary key. When the
        column is absent from ``fields`` it is auto-generated as sequential
        integers ``0 .. count - 1``; otherwise the field provider must produce
        unique values.
    """

    name: str
    count: int = 0
    fields: Dict[str, Provider] = field(default_factory=dict)
    primary_key: str = "id"


@dataclass
class EntitySpec:
    """An entity paired with its outgoing references."""

    entity: Entity
    references: List[Reference] = field(default_factory=list)


def generate_dataset(
    entities: Sequence[Union[Entity, EntitySpec]],
    faker: Any,
    seed: SeedType = 0,
) -> Dict[str, List[Dict[str, Any]]]:
    """Generate a complete dataset satisfying all declared references.

    Definitions and cardinality feasibility are validated before any row is
    built, so the result is either a fully consistent dataset or an exception
    describing the offending relation -- never a half-populated dataset.

    :param entities: entity descriptions. An :class:`Entity` without references
        may be passed directly; use :class:`EntitySpec` to attach references.
    :param faker: configured ``Faker`` instance used for regular columns.
    :param seed: seed for the internal RNG that assigns references. It is kept
        separate from the supplied Faker instance, which is not re-seeded.
    """

    specs = _as_specs(entities)
    _validate_specs(specs)
    _check_feasibility(specs)

    rng = random_module.Random()
    rng.seed(seed)

    by_name = {spec.entity.name: spec for spec in specs}
    dataset: Dict[str, List[Dict[str, Any]]] = {}
    for spec in specs:
        rows = [{column: None for column in _all_columns(spec)} for _ in range(spec.entity.count)]
        _fill_primary_keys(spec, rows, faker)
        _fill_fields(spec, rows, faker)
        dataset[spec.entity.name] = rows

    for spec in specs:
        target_spec = None
        for reference in spec.references:
            target_spec = by_name[reference.target]
            _assign_reference(
                reference,
                spec.entity.name,
                target_spec.entity.primary_key,
                dataset[spec.entity.name],
                dataset[reference.target],
                rng,
            )

    return dataset


def _as_specs(entities: Sequence[Union[Entity, EntitySpec]]) -> List[EntitySpec]:
    specs: List[EntitySpec] = []
    for item in entities:
        if isinstance(item, EntitySpec):
            specs.append(item)
        elif isinstance(item, Entity):
            specs.append(EntitySpec(item))
        else:
            raise RelationDefinitionError(f"Expected Entity or EntitySpec, got {type(item).__name__}")
    return specs


def _validate_specs(specs: Sequence[EntitySpec]) -> None:
    names = [spec.entity.name for spec in specs]
    if not names:
        raise RelationDefinitionError("At least one entity must be declared")
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise RelationDefinitionError(f"Duplicate entity names: {', '.join(duplicates)}")

    by_name = {spec.entity.name: spec for spec in specs}

    for spec in specs:
        entity = spec.entity
        label = entity.name

        if not isinstance(entity.count, int) or isinstance(entity.count, bool) or entity.count < 0:
            raise RelationDefinitionError(
                f"Entity '{label}' count must be a non-negative integer, got {entity.count!r}"
            )
        if not isinstance(entity.primary_key, str) or not entity.primary_key:
            raise RelationDefinitionError(f"Entity '{label}' primary key must be a non-empty string")

        for column, provider in entity.fields.items():
            if not isinstance(column, str) or not column:
                raise RelationDefinitionError(f"Entity '{label}' has an empty field name")
            if not (isinstance(provider, str) or callable(provider)):
                raise RelationDefinitionError(
                    f"Field '{label}.{column}' provider must be a formatter name or zero-argument callable"
                )

        for reference in spec.references:
            if not isinstance(reference, Reference):
                raise RelationDefinitionError(f"Reference on entity '{label}' must be a Reference instance")
            ref_label = f"{label}.{reference.field}"
            if not reference.field:
                raise RelationDefinitionError(f"Entity '{label}' has a reference with an empty field name")
            if reference.field == entity.primary_key:
                raise RelationDefinitionError(f"Reference '{ref_label}' cannot reuse the primary-key column")
            if reference.field in entity.fields:
                raise RelationDefinitionError(f"Reference '{ref_label}' clashes with a regular field of the same name")
            if reference.target not in by_name:
                raise RelationDefinitionError(
                    f"Reference '{ref_label}' targets unknown entity '{reference.target}'"
                )
            if not 0.0 <= reference.null_fraction <= 1.0:
                raise RelationDefinitionError(
                    f"Reference '{ref_label}' null_fraction must be between 0 and 1, got {reference.null_fraction}"
                )


def _check_feasibility(specs: Sequence[EntitySpec]) -> None:
    by_name = {spec.entity.name: spec for spec in specs}
    for spec in specs:
        source_count = spec.entity.count
        for reference in spec.references:
            target_count = by_name[reference.target].entity.count
            label = f"{spec.entity.name}.{reference.field}"
            if reference.nullable:
                # Nullable references are always feasible: rows without an
                # available target are simply left as None.
                continue
            if target_count == 0:
                raise UnsatisfiableRelationError(
                    f"Required reference '{label}' needs at least 1 "
                    f"'{reference.target}' row but none are generated",
                    reference=label,
                    required_size=1,
                )
            if reference.unique and source_count > target_count:
                raise UnsatisfiableRelationError(
                    f"Required one-to-one reference '{label}' needs at least {source_count} distinct "
                    f"'{reference.target}' rows but only {target_count} are generated",
                    reference=label,
                    required_size=source_count,
                )


def _all_columns(spec: EntitySpec) -> List[str]:
    columns = [spec.entity.primary_key]
    columns.extend(column for column in spec.entity.fields if column != spec.entity.primary_key)
    columns.extend(reference.field for reference in spec.references)
    return columns


def _fill_primary_keys(spec: EntitySpec, rows: List[Dict[str, Any]], faker: Any) -> None:
    entity = spec.entity
    if entity.primary_key not in entity.fields:
        for index, row in enumerate(rows):
            row[entity.primary_key] = index
        return

    values = [_call_provider(entity.fields[entity.primary_key], faker) for _ in rows]
    if len({_hash_key(value) for value in values}) != len(values):
        raise RelationDefinitionError(
            f"Primary-key provider for entity '{entity.name}' returned duplicate values"
        )
    for row, value in zip(rows, values):
        row[entity.primary_key] = value


def _fill_fields(spec: EntitySpec, rows: List[Dict[str, Any]], faker: Any) -> None:
    for column, provider in spec.entity.fields.items():
        if column == spec.entity.primary_key:
            continue
        for row in rows:
            row[column] = _call_provider(provider, faker)


def _call_provider(provider: Provider, faker: Any) -> Any:
    if isinstance(provider, str):
        return faker.format(provider)
    return provider()


def _hash_key(value: Any) -> Any:
    try:
        hash(value)
        return value
    except TypeError:
        return repr(value)


def _assign_reference(
    reference: Reference,
    source_name: str,
    target_pk: str,
    source_rows: List[Dict[str, Any]],
    target_rows: List[Dict[str, Any]],
    rng: random_module.Random,
) -> None:
    target_keys = [row[target_pk] for row in target_rows]

    if not target_keys:
        # Feasibility guarantees this only happens for nullable references.
        for row in source_rows:
            row[reference.field] = None
        return

    if reference.nullable:
        non_null_indices = [
            index for index, _ in enumerate(source_rows) if rng.random() >= reference.null_fraction
        ]
        if reference.unique and len(non_null_indices) > len(target_keys):
            rng.shuffle(non_null_indices)
            non_null_indices = non_null_indices[: len(target_keys)]
        null_indices = set(range(len(source_rows))) - set(non_null_indices)
        for index in null_indices:
            source_rows[index][reference.field] = None

        if reference.unique:
            keys = list(target_keys)
            rng.shuffle(keys)
            assignments = zip(non_null_indices, keys)
        else:
            assignments = ((index, rng.choice(target_keys)) for index in non_null_indices)
        for index, key in assignments:
            source_rows[index][reference.field] = key
        return

    if reference.unique:
        keys = list(target_keys)
        rng.shuffle(keys)
        for row, key in zip(source_rows, keys):
            row[reference.field] = key
    else:
        for row in source_rows:
            row[reference.field] = rng.choice(target_keys)
