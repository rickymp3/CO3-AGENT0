"""Minimal RC Trajectory toy system.
Runs a discrete trajectory over a finite grid, logs feasibility/margin/coherence/trend.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple


@dataclass(frozen=True)
class Action:
    name: str
    delta: Tuple[int, int]


@dataclass
class Config:
    grid_size: Tuple[int, int] = (6, 6)  # (width, height)
    obstacles: Tuple[Tuple[int, int], ...] = ((2, 3), (3, 1), (4, 4))
    start: Tuple[int, int] = (1, 1)
    steps: int = 18
    window: int = 3
    dt: float = 1.0
    lam: float = 1.0
    mu: float = 0.4


@dataclass
class StepRecord:
    step: int
    state: Tuple[int, int]
    action: str
    feasible_size: int
    selected_margin: float
    min_margin: float
    coherence: float
    trend: float
    boundary_crossed: bool


ACTIONS: Sequence[Action] = (
    Action("stay", (0, 0)),
    Action("east", (1, 0)),
    Action("north", (0, 1)),
    Action("west", (-1, 0)),
    Action("south", (0, -1)),
)


def in_bounds(pos: Tuple[int, int], cfg: Config) -> bool:
    x, y = pos
    return 0 <= x < cfg.grid_size[0] and 0 <= y < cfg.grid_size[1]


def is_obstacle(pos: Tuple[int, int], cfg: Config) -> bool:
    return pos in cfg.obstacles


def is_boundary(pos: Tuple[int, int], cfg: Config) -> bool:
    return not in_bounds(pos, cfg) or is_obstacle(pos, cfg)


def margin_to_boundary(pos: Tuple[int, int], cfg: Config) -> float:
    """Manhattan distance to nearest boundary (grid edge or obstacle)."""
    if is_boundary(pos, cfg):
        return 0.0

    x, y = pos
    width, height = cfg.grid_size
    # Distance to outer boundary cells.
    dist_edge = min(x, y, width - 1 - x, height - 1 - y)

    # Distance to obstacle surfaces (0 if adjacent).
    dist_obstacles = []
    for ox, oy in cfg.obstacles:
        dist = max(abs(x - ox) + abs(y - oy) - 1, 0)
        dist_obstacles.append(dist)
    dist_obstacle = min(dist_obstacles) if dist_obstacles else float("inf")

    return float(min(dist_edge, dist_obstacle))


def feasible_options(state: Tuple[int, int], cfg: Config) -> Dict[Action, float]:
    """Return feasible actions mapped to their margin."""
    options: Dict[Action, float] = {}
    for action in ACTIONS:
        nx, ny = state[0] + action.delta[0], state[1] + action.delta[1]
        candidate = (nx, ny)
        if is_boundary(candidate, cfg):
            continue
        options[action] = margin_to_boundary(candidate, cfg)
    return options


def select_action(options: Dict[Action, float], prev_action: Action | None, cfg: Config) -> Action:
    """Pick action minimizing objective: lambda/margin + mu*switch_penalty."""
    best_action: Action | None = None
    best_score = float("inf")
    for action, margin in options.items():
        switch_penalty = 0.0 if prev_action and action.name == prev_action.name else 1.0
        score = cfg.lam * (1.0 / max(margin, 1e-6)) + cfg.mu * switch_penalty
        if score < best_score:
            best_score = score
            best_action = action
    assert best_action is not None
    return best_action


def coherence(actions: Sequence[str], window: int) -> float:
    if not actions:
        return 0.0
    tail = actions[-window:]
    counts: Dict[str, int] = {}
    for name in tail:
        counts[name] = counts.get(name, 0) + 1
    mode_count = max(counts.values())
    return mode_count / len(tail)


def trend(actions: Sequence[str], window: int, dt: float) -> float:
    if len(actions) < 2 * window:
        return 0.0
    recent = actions[-window:]
    prev = actions[-2 * window : -window]
    return (coherence(recent, window) - coherence(prev, window)) / (window * dt)


def step_state(state: Tuple[int, int], action: Action) -> Tuple[int, int]:
    return state[0] + action.delta[0], state[1] + action.delta[1]


def run(cfg: Config) -> List[StepRecord]:
    records: List[StepRecord] = []
    state = cfg.start
    actions_taken: List[str] = []
    prev_action: Action | None = None

    for step in range(cfg.steps):
        options = feasible_options(state, cfg)
        if not options:
            records.append(
                StepRecord(
                    step=step,
                    state=state,
                    action="none",
                    feasible_size=0,
                    selected_margin=0.0,
                    min_margin=0.0,
                    coherence=coherence(actions_taken, cfg.window),
                    trend=trend(actions_taken, cfg.window, cfg.dt),
                    boundary_crossed=True,
                )
            )
            break

        chosen = select_action(options, prev_action, cfg)
        next_state = step_state(state, chosen)

        actions_taken.append(chosen.name)
        coh = coherence(actions_taken, cfg.window)
        tr = trend(actions_taken, cfg.window, cfg.dt)

        margin_values = list(options.values())
        records.append(
            StepRecord(
                step=step,
                state=state,
                action=chosen.name,
                feasible_size=len(options),
                selected_margin=options[chosen],
                min_margin=min(margin_values),
                coherence=coh,
                trend=tr,
                boundary_crossed=False,
            )
        )

        state = next_state
        prev_action = chosen

    return records


def summarize(records: Sequence[StepRecord]) -> None:
    print("step\tstate\taction\tfeasible\tmargin\tmin_margin\tcoherence\ttrend")
    for r in records:
        print(
            f"{r.step}\t{r.state}\t{r.action}\t{r.feasible_size}\t"
            f"{r.selected_margin:.2f}\t{r.min_margin:.2f}\t{r.coherence:.2f}\t{r.trend:.3f}"
        )
    collapsed = any(r.feasible_size == 0 for r in records)
    print(f"boundary_crossed: {collapsed}")


def main() -> None:
    base_cfg = Config()
    alt_cfg = Config(start=(3, 2))

    print("=== Run A (start=(1,1)) ===")
    records_a = run(base_cfg)
    summarize(records_a)
    print()

    print("=== Run B (start=(3,2)) ===")
    records_b = run(alt_cfg)
    summarize(records_b)


if __name__ == "__main__":
    main()
