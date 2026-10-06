"""Generate small relational datasets on top of regular Faker providers.

This module lets a caller declare a set of entities (row counts, field
providers, primary keys) plus one-to-one or many-to-one references between
them, and produces a complete in-memory dataset that satisfies those
relations.  It never touches a database and never evaluates arbitrary code:
plain fields are produced exclusively by provider *names* resolved through
the usual ``Generator.format`` machinery, and references are described with
structured :class:`Ref` declarations only.

Generation is atomic: :meth:`RelationalDatasetGenerator.generate` either
returns one whole dataset in which every declared relation holds, or raises
:class:`UnsatisfiableRelationError` (cardinality makes a relation impossible)
or :class:`RelationalSchemaError` (the declaration itself is invalid).  No
partial dataset is ever returned or stored, so a failed run cannot leave
behind half-generated output that might be misused.
"""

from dataclasses import dataclass, field as dataclass_field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..exceptions import BaseFakerException
from ..proxy import Faker


class RelationalSchemaError(BaseFakerException):
    """The relational dataset declaration itself is invalid."""


class UnsatisfiableRelationError(BaseFakerException):
    """A declared relation cannot be satisfied given the entity counts."""


@dataclass(frozen=True)
class Ref:
    """A many-to-one (or one-to-one, when ``unique=True``) reference.

    ``field`` is the foreign-key column created on the source entity,
    ``target`` the name of the referenced entity and ``target_field`` the
    referenced column (defaults to the target's primary key).  A ``required``
    reference is never ``None``; an optional one may be.  ``unique`` makes
    the relation one-to-one: no two source rows may point at the same target
    value.  ``allow_self`` only matters for self-references (``target`` is
    the source entity itself); when ``False`` a row may not reference its
    own primary key.
    """

    field: str
    target: str
    target_field: Optional[str] = None
    required: bool = True
    unique: bool = False
    allow_self: bool = True


@dataclass(frozen=True)
class Entity:
    """One entity (table) of the relational dataset.

    ``fields`` maps column names to Faker provider names (e.g. ``"name"``,
    ``"email"``).  If the primary key ``pk`` is not listed in ``fields``,
    sequential integers starting at 1 are assigned automatically; if it is
    listed, the provider's values are used and must come out unique.
    """

    name: str
    count: int
    pk: str
    fields: Mapping[str, str] = dataclass_field(default_factory=dict)
    refs: Tuple[Ref, ...] = ()


class RelationalDatasetGenerator:
    """Generates whole relational datasets from :class:`Entity` declarations."""

    def __init__(self, faker: Optional[Faker] = None, seed: Optional[int] = None) -> None:
        self.faker = faker if faker is not None else Faker()
        if seed is not None:
            self.faker.seed_instance(seed)

    def generate(self, entities: Sequence[Entity]) -> Dict[str, List[Dict[str, Any]]]:
        """Return ``{entity_name: [row, ...]}`` with every relation satisfied.

        Raises :class:`RelationalSchemaError` for invalid declarations and
        :class:`UnsatisfiableRelationError` when some relation cannot be
        satisfied with the declared counts.  Either a complete dataset is
        returned or an exception is raised; nothing partial is exposed.
        """
        entities = tuple(entities)
        by_name = self._validate_schema(entities)
        self._check_satisfiability(entities, by_name)
        rows = self._generate_base_rows(entities)
        self._resolve_references(entities, by_name, rows)
        self._verify(entities, by_name, rows)
        return {name: list(entity_rows) for name, entity_rows in rows.items()}

    @staticmethod
    def _relation_label(entity: Entity, ref: Ref, target_field: str) -> str:
        return f"{entity.name}.{ref.field} -> {ref.target}.{target_field}"

    def _validate_schema(self, entities: Tuple[Entity, ...]) -> Dict[str, Entity]:
        by_name: Dict[str, Entity] = {}
        for entity in entities:
            if entity.name in by_name:
                raise RelationalSchemaError(f"Duplicate entity name {entity.name!r}")
            if entity.count < 0:
                raise RelationalSchemaError(f"Entity {entity.name!r} has a negative count")
            if not entity.pk:
                raise RelationalSchemaError(f"Entity {entity.name!r} declares no primary key")
            by_name[entity.name] = entity

        for entity in entities:
            for column, provider in entity.fields.items():
                if not isinstance(provider, str) or provider.startswith("_"):
                    raise RelationalSchemaError(
                        f"Field {entity.name}.{column} must name a Faker provider, got {provider!r}"
                    )
                try:
                    self.faker.get_formatter(provider)
                except AttributeError:
                    raise RelationalSchemaError(
                        f"Field {entity.name}.{column} refers to unknown provider {provider!r}"
                    ) from None

            for ref in entity.refs:
                if ref.field == entity.pk or ref.field in entity.fields:
                    raise RelationalSchemaError(
                        f"Reference field {entity.name}.{ref.field} collides with a declared field"
                    )
                target = by_name.get(ref.target)
                if target is None:
                    raise RelationalSchemaError(
                        f"Relation {entity.name}.{ref.field} targets unknown entity {ref.target!r}"
                    )
                target_field = ref.target_field or target.pk
                if target_field != target.pk and target_field not in target.fields:
                    raise RelationalSchemaError(
                        f"Relation {entity.name}.{ref.field} targets unknown column "
                        f"{ref.target}.{target_field}"
                    )
        return by_name

    def _check_satisfiability(self, entities: Tuple[Entity, ...], by_name: Dict[str, Entity]) -> None:
        for entity in entities:
            for ref in entity.refs:
                target = by_name[ref.target]
                label = self._relation_label(entity, ref, ref.target_field or target.pk)
                if ref.required and target.count == 0:
                    raise UnsatisfiableRelationError(
                        f"Relation {label} is required but entity {ref.target!r} has no rows"
                    )
                if ref.required and ref.unique and entity.count > target.count:
                    raise UnsatisfiableRelationError(
                        f"Relation {label} is unique and required: {entity.count} rows of "
                        f"{entity.name!r} cannot each reference a distinct row of "
                        f"{ref.target!r}, which only has {target.count}"
                    )
                if ref.required and not ref.allow_self and ref.target == entity.name and entity.count == 1:
                    raise UnsatisfiableRelationError(
                        f"Relation {label} forbids self-references but the single row of "
                        f"{entity.name!r} has nothing else to reference"
                    )

    def _generate_base_rows(self, entities: Tuple[Entity, ...]) -> Dict[str, List[Dict[str, Any]]]:
        rows: Dict[str, List[Dict[str, Any]]] = {}
        for entity in entities:
            entity_rows: List[Dict[str, Any]] = []
            for index in range(entity.count):
                row = {column: self.faker.format(provider) for column, provider in entity.fields.items()}
                if entity.pk not in entity.fields:
                    row[entity.pk] = index + 1
                entity_rows.append(row)
            pk_values = [row[entity.pk] for row in entity_rows]
            if len(set(pk_values)) != len(pk_values):
                raise RelationalSchemaError(
                    f"Primary key {entity.name}.{entity.pk} produced duplicate values; "
                    f"use a provider that yields unique values or omit it from 'fields' "
                    f"to get sequential ids"
                )
            rows[entity.name] = entity_rows
        return rows

    def _resolve_references(
        self,
        entities: Tuple[Entity, ...],
        by_name: Dict[str, Entity],
        rows: Dict[str, List[Dict[str, Any]]],
    ) -> None:
        random = self.faker.random
        for entity in entities:
            entity_rows = rows[entity.name]
            for ref in entity.refs:
                target = by_name[ref.target]
                target_field = ref.target_field or target.pk
                pool = [row[target_field] for row in rows[ref.target]]
                self_ref = ref.target == entity.name

                if ref.unique:
                    values = list(dict.fromkeys(pool))
                    random.shuffle(values)
                    if ref.required:
                        if self_ref and not ref.allow_self:
                            # Rotating a shuffled permutation by one yields a
                            # derangement: no row points at its own primary key.
                            assignments = [values[(i + 1) % len(values)] for i in range(len(entity_rows))]
                        else:
                            assignments = values[: len(entity_rows)]
                        for row, value in zip(entity_rows, assignments):
                            row[ref.field] = value
                    else:
                        for row in entity_rows:
                            if values and random.random() < 0.5:
                                row[ref.field] = values.pop()
                            else:
                                row[ref.field] = None
                else:
                    for row in entity_rows:
                        candidates = pool
                        if self_ref and not ref.allow_self:
                            own = row[entity.pk]
                            candidates = [value for value in pool if value != own]
                        if not candidates or (not ref.required and random.random() < 0.5):
                            row[ref.field] = None
                        else:
                            row[ref.field] = random.choice(candidates)

    def _verify(
        self,
        entities: Tuple[Entity, ...],
        by_name: Dict[str, Entity],
        rows: Dict[str, List[Dict[str, Any]]],
    ) -> None:
        for entity in entities:
            for ref in entity.refs:
                target = by_name[ref.target]
                target_field = ref.target_field or target.pk
                label = self._relation_label(entity, ref, target_field)
                valid_values = {row[target_field] for row in rows[ref.target]}
                seen = set()
                for row in rows[entity.name]:
                    value = row[ref.field]
                    if value is None:
                        if ref.required:
                            raise UnsatisfiableRelationError(f"Relation {label} produced a missing value")
                        continue
                    if value not in valid_values:
                        raise UnsatisfiableRelationError(
                            f"Relation {label} produced a value absent from {ref.target}.{target_field}"
                        )
                    if ref.unique:
                        if value in seen:
                            raise UnsatisfiableRelationError(f"Relation {label} produced a duplicate value")
                        seen.add(value)
                    if ref.target == entity.name and not ref.allow_self and value == row[entity.pk]:
                        raise UnsatisfiableRelationError(f"Relation {label} produced a self-reference")


def generate_dataset(
    entities: Sequence[Entity],
    seed: Optional[int] = None,
    faker: Optional[Faker] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Convenience wrapper around :class:`RelationalDatasetGenerator`."""
    return RelationalDatasetGenerator(faker=faker, seed=seed).generate(entities)
