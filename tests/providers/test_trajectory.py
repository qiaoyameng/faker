import pytest

from faker import Faker
from faker.providers import TrajectoryEvent


@pytest.fixture
def fake():
    fake = Faker()
    fake.seed_instance(0)
    return fake


def forced_path_probability(fake, monkeypatch, transitions, path, **kwargs):
    selected = iter(path[1:])
    branch_probabilities = []

    def deterministic_choices(elements, weights, random, length):
        destination = next(selected)
        index = elements.index(destination)
        branch_probabilities.append(weights[index] / sum(weights))
        return [destination]

    monkeypatch.setattr("faker.providers.choices_distribution", deterministic_choices)
    generated = tuple(fake.random_trajectory(transitions, **kwargs))
    assert generated == path

    probability = 1.0
    for branch_probability in branch_probabilities:
        probability *= branch_probability
    return probability


def test_trajectory_follows_original_model_conditioned_on_terminal(fake, monkeypatch):
    transitions = {
        "created": {"paid": 3, "cancelled": 1},
        "paid": {"shipped": 2, "refunded": 1},
        "shipped": {"completed": 1},
        "cancelled": {"cancelled": 1},
    }
    common_arguments = {
        "start_state": "created",
        "end_states": ("refunded", "cancelled"),
        "steps": 2,
    }

    refunded_probability = forced_path_probability(
        fake,
        monkeypatch,
        transitions,
        ("created", "paid", "refunded"),
        **common_arguments,
    )
    cancelled_probability = forced_path_probability(
        fake,
        monkeypatch,
        transitions,
        ("created", "cancelled", "cancelled"),
        **common_arguments,
    )

    assert refunded_probability == 3 / 4
    assert cancelled_probability == 1 / 4

    completed_probability = forced_path_probability(
        fake,
        monkeypatch,
        transitions,
        ("created", "paid", "shipped", "completed"),
        start_state="created",
        end_states={"completed"},
        steps=3,
    )
    assert completed_probability == 1.0


def test_cycles_preserve_path_weight_and_do_not_create_infinite_paths(fake, monkeypatch):
    transitions = {
        "created": {"created": 1, "paid": 2},
        "paid": {"shipped": 1, "paid": 3},
        "shipped": {"completed": 4},
        "completed": {},
    }
    common_arguments = {
        "start_state": "created",
        "end_states": {"completed"},
        "steps": 4,
    }

    start_loop_probability = forced_path_probability(
        fake,
        monkeypatch,
        transitions,
        ("created", "created", "paid", "shipped", "completed"),
        **common_arguments,
    )
    paid_loop_probability = forced_path_probability(
        fake,
        monkeypatch,
        transitions,
        ("created", "paid", "paid", "shipped", "completed"),
        **common_arguments,
    )

    assert start_loop_probability == 8 / 32
    assert paid_loop_probability == 24 / 32

    with pytest.raises(ValueError, match="No trajectory"):
        fake.state_trajectory(
            transitions,
            start_state="created",
            end_states={"completed"},
            steps=2,
        )


def test_edges_that_become_dead_have_zero_conditional_weight(fake, monkeypatch):
    transitions = {
        "created": {"paid": 1},
        "paid": {"shipped": 1, "refunded": 1},
        "shipped": {"completed": 1, "cancelled": 1},
        "cancelled": {"cancelled": 1},
        "refunded": {"refunded": 1},
        "completed": {},
    }

    selected = iter(["paid", "shipped", "completed"])
    sampled_weights = []

    def deterministic_choices(elements, weights, random, length):
        sampled_weights.append(list(zip(elements, weights)))
        return [next(selected)]

    monkeypatch.setattr("faker.providers.choices_distribution", deterministic_choices)
    path = fake.random_trajectory(
        transitions,
        start_state="created",
        end_states="completed",
        steps=3,
    )

    assert path == ["created", "paid", "shipped", "completed"]
    assert sampled_weights == [
        [("paid", 1.0)],
        [("shipped", 1.0), ("refunded", 0.0)],
        [("completed", 1.0), ("cancelled", 0.0)],
    ]


def test_impossible_lengths_fail_before_random_selection(fake):
    transitions = {
        "created": {"paid": 1},
        "paid": {},
    }
    calls = []

    def choices(elements, weights, random, length):
        calls.append((elements, weights))
        return [elements[0]]

    fake.random.uniform = lambda start, stop: calls.append((start, stop)) or start
    fake.seed_instance(1)

    with pytest.raises(ValueError, match="No trajectory"):
        fake.random_trajectory(
            transitions,
            start_state="created",
            end_states="paid",
            steps=2,
        )
    assert calls == []


def test_zero_weight_edges_and_absorbing_states(fake):
    transitions = [
        ("created", "paid", 0),
        ("created", "cancelled", 1),
        ("cancelled", "cancelled", 1),
        ("paid", "shipped", 1),
    ]
    for _ in range(50):
        path = fake.random_trajectory(
            transitions,
            start_state="created",
            end_states=("cancelled",),
            steps=2,
        )
        assert path == ["created", "cancelled", "cancelled"]

    with pytest.raises(ValueError, match="No trajectory"):
        fake.random_trajectory(
            transitions,
            start_state="created",
            end_states="shipped",
            steps=2,
        )


def test_duration_sampler_is_independent_and_non_negative(fake, monkeypatch):
    transitions = {"created": {"paid": 1}, "paid": {"completed": 1}}
    durations = iter([0, 2.5])

    monkeypatch.setattr(
        "faker.providers.choices_distribution",
        lambda elements, weights, random, length: [elements[0]],
    )
    trajectory = fake.random_trajectory(
        transitions,
        start_state="created",
        end_states="completed",
        steps=2,
        duration_sampler=lambda: next(durations),
    )

    assert trajectory.states == ["created", "paid", "completed"]
    assert trajectory.events == [
        TrajectoryEvent("created", 0, 0),
        TrajectoryEvent("paid", 0, 0),
        TrajectoryEvent("completed", 2.5, 2.5),
    ]
    assert trajectory.durations == [0, 0, 2.5]
    assert trajectory.timestamps == [0, 0, 2.5]

    with pytest.raises(ValueError, match="non-negative"):
        fake.random_trajectory(
            transitions,
            start_state="created",
            end_states="paid",
            steps=1,
            duration_sampler=lambda: -0.1,
        )


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"start_state": "missing", "end_states": "paid", "steps": 1}, "No trajectory"),
        ({"start_state": "created", "end_states": "paid", "steps": -1}, "non-negative integer"),
        ({"start_state": "created", "end_states": (), "steps": 1}, "at least one"),
    ],
)
def test_invalid_trajectory_arguments(fake, kwargs, match):
    transitions = {"created": {"paid": 1}, "paid": {}}
    with pytest.raises(ValueError, match=match):
        fake.random_trajectory(transitions, **kwargs)


def test_zero_steps_requires_start_to_be_allowed_terminal(fake):
    transitions = {"created": {"paid": 1}, "paid": {}}
    assert fake.random_trajectory(
        transitions,
        start_state="paid",
        end_states="paid",
        steps=0,
    ) == ["paid"]
    with pytest.raises(ValueError, match="No trajectory"):
        fake.random_trajectory(
            transitions,
            start_state="created",
            end_states="paid",
            steps=0,
        )


def test_adjacency_sequence_transitions_and_invalid_weights(fake):
    transitions = {
        "created": [("paid", 1)],
        "paid": [("completed", 2), ("refunded", 0)],
        "completed": [],
        "refunded": [],
    }
    for _ in range(20):
        assert fake.trajectory(
            transitions,
            start_state="created",
            end_states="completed",
            steps=2,
        ) == ["created", "paid", "completed"]

    transitions["paid"] = [("completed", -1)]
    with pytest.raises(ValueError, match="finite, non-negative"):
        fake.trajectory(
            transitions,
            start_state="created",
            end_states="completed",
            steps=2,
        )


def test_duration_sampler_does_not_consume_path_randomness(fake, monkeypatch):
    transitions = {"created": {"paid": 1}, "paid": {"completed": 1}}
    path_draws = []

    def deterministic_choices(elements, weights, random, length):
        path_draws.append(random.random())
        return [elements[0]]

    monkeypatch.setattr("faker.providers.choices_distribution", deterministic_choices)
    fake.seed_instance(7)
    fake.trajectory(
        transitions,
        start_state="created",
        end_states="completed",
        steps=2,
        duration_sampler=fake.random.random,
    )

    fake.seed_instance(7)
    expected_draws = [fake.random.random(), fake.random.random()]
    assert path_draws == expected_draws
