"""ASCII-only simulation for prompt-based metric convergence.

This CLI runs a deterministic loop that generates neutral English prompts,
feeds them through a toy scoring function, updates normalized variables
(S, K, P, B, H) toward 1.00, and prints a final report once all variables
converge. No networking or external APIs are used.
"""
from __future__ import annotations

import argparse
import random
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

VARIABLES = ("S", "K", "P", "B", "H")


@dataclass
class SimulationState:
    values: Dict[str, float] = field(default_factory=lambda: {k: 0.0 for k in VARIABLES})
    turn: int = 0
    prompts: List[str] = field(default_factory=list)

    def snapshot(self) -> Dict[str, float]:
        return dict(self.values)


def build_prompt(rng: random.Random) -> str:
    subjects = [
        "How does the summary remain consistent?",
        "What supports the key claim in this run?",
        "Which pattern still holds under variation?",
        "Where do boundary cases challenge the model?",
        "Why is the hypothesis stable across trials?",
    ]
    connectors = [
        "Please explain in plain terms.",
        "Provide a concise check.",
        "State the simplest evidence.",
        "Offer a short verification.",
        "Give a neutral review.",
    ]
    endings = [
        "Keep it brief and factual.",
        "Focus on clarity.",
        "Stay objective.",
        "Avoid speculation.",
        "Ensure consistency.",
    ]

    return " ".join(
        [
            rng.choice(subjects),
            rng.choice(connectors),
            rng.choice(endings),
        ]
    )


def generate_prompts(rng: random.Random, count: int) -> List[str]:
    return [build_prompt(rng) for _ in range(count)]


def score_prompts(prompts: Sequence[str], rng: random.Random) -> Dict[str, float]:
    length_signal = sum(len(p) for p in prompts) / max(1, len(prompts))
    scores: Dict[str, float] = {}
    for idx, name in enumerate(VARIABLES):
        base = 0.35 + 0.05 * idx
        noise = 0.15 * rng.random()
        prompt_term = (length_signal % (idx + 5)) / (idx + 5)
        score = min(1.0, base + noise + prompt_term)
        scores[name] = score
    return scores


def update_variables(values: Dict[str, float], scores: Dict[str, float]) -> Dict[str, float]:
    updated: Dict[str, float] = {}
    for name, current in values.items():
        lift = 0.4 * (1.0 - current)
        contribution = 0.2 * scores.get(name, 0.0)
        updated[name] = min(1.0, current + lift + contribution)
    return updated


def all_reached(values: Dict[str, float]) -> bool:
    return all(value >= 1.0 for value in values.values())


def run_simulation(seed: int, prompt_count: int, max_turns: int) -> List[SimulationState]:
    history: List[SimulationState] = []
    state = SimulationState()

    for turn in range(1, max_turns + 1):
        turn_seed = (seed << 8) + turn
        rng = random.Random(turn_seed)
        prompts = generate_prompts(rng, prompt_count)
        scores = score_prompts(prompts, rng)
        state.values = update_variables(state.values, scores)
        state.turn = turn
        state.prompts = prompts
        history.append(SimulationState(values=state.snapshot(), turn=turn, prompts=list(prompts)))
        if all_reached(state.values):
            break

    return history


def format_report(history: Sequence[SimulationState], seed: int) -> str:
    if not history:
        return "No turns executed."

    lines: List[str] = []
    lines.append("ASCII Metric Convergence Report")
    lines.append(f"Seed: {seed}")
    lines.append(f"Total turns: {history[-1].turn}")
    lines.append("")
    lines.append("Turn\tS\tK\tP\tB\tH")
    for entry in history:
        values = entry.values
        lines.append(
            f"{entry.turn}\t"
            f"{values['S']:.2f}\t{values['K']:.2f}\t{values['P']:.2f}\t"
            f"{values['B']:.2f}\t{values['H']:.2f}"
        )

    lines.append("")
    lines.append("Final turn prompts:")
    for prompt in history[-1].prompts:
        lines.append(f"- {prompt}")

    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an ASCII-only prompt simulation until all variables reach 1.00.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seed", type=int, default=1234, help="Base seed for deterministic randomness")
    parser.add_argument("--prompt-count", type=int, default=3, help="Number of prompts per turn")
    parser.add_argument("--max-turns", type=int, default=50, help="Maximum turns before stopping")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    history = run_simulation(seed=args.seed, prompt_count=args.prompt_count, max_turns=args.max_turns)
    report = format_report(history, seed=args.seed)
    print(report)


if __name__ == "__main__":
    main()
