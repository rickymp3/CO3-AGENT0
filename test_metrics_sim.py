"""Tests for the ASCII prompt convergence simulation."""

import random

from metrics_sim import all_reached, generate_prompts, run_simulation


def test_generate_prompts_is_deterministic():
    seed = 99
    rng_a = random.Random(seed)
    rng_b = random.Random(seed)

    prompts_a = generate_prompts(rng_a, 3)
    prompts_b = generate_prompts(rng_b, 3)

    assert prompts_a == prompts_b


def test_run_simulation_reaches_one():
    history = run_simulation(seed=7, prompt_count=2, max_turns=10)

    assert history, "Simulation should record at least one turn"
    final_state = history[-1]
    assert all_reached(final_state.values)
    # Ensure determinism for this seed
    assert final_state.turn == 3
    assert final_state.values == {
        "S": 1.0,
        "K": 1.0,
        "P": 1.0,
        "B": 1.0,
        "H": 1.0,
    }
