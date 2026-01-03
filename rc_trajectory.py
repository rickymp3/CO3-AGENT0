"""Minimal RC Trajectory toy system.
Runs a discrete trajectory over a finite grid, logs feasibility/margin/coherence/trend.
"""
from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from getpass import getpass
from typing import Callable, Dict, List, Sequence, Tuple


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


@dataclass
class CommunicationWrapper:
    """Bidirectional mapper for human-facing signals and model variables.

    The wrapper now includes a mediation step that parses human text, normalizes
    numeric values onto a 0.0–1.0 scale, passes the mapped variables to a model,
    and rewraps the model's structured response back into English for the user.

    Keys that are absent from the provided mappings are passed through
    unchanged, so the wrapper can be used incrementally.
    """

    input_mapping: Dict[str, str]
    output_mapping: Dict[str, str]
    variable_ranges: Dict[str, Tuple[float, float]] | None = None

    def encode_input(self, signal: Dict[str, object]) -> Dict[str, object]:
        """Map a human input signal into model-facing variables."""

        mapped: Dict[str, object] = {}
        for human_key, value in signal.items():
            model_key = self.input_mapping.get(human_key, human_key)
            mapped[model_key] = value
        return mapped

    def decode_output(self, response: Dict[str, object]) -> Dict[str, object]:
        """Map a model response into human-facing variables."""

        mapped: Dict[str, object] = {}
        for model_key, value in response.items():
            human_key = self.output_mapping.get(model_key, model_key)
            mapped[human_key] = value
        return mapped

    def _normalize_value(self, key: str, value: object) -> object:
        """Normalize numeric signals to 0.0–1.0 using optional ranges."""

        if not isinstance(value, (int, float)):
            return value

        ranges = self.variable_ranges or {}
        low, high = ranges.get(key, (0.0, 1.0))
        if high == low:
            return 0.0

        normalized = (float(value) - low) / (high - low)
        return max(0.0, min(1.0, normalized))

    def normalize_signal(self, signal: Dict[str, object]) -> Dict[str, object]:
        """Apply numeric normalization to all mapped variables."""

        normalized: Dict[str, object] = {}
        for key, value in signal.items():
            normalized[key] = self._normalize_value(key, value)
        return normalized

    def parse_user_signal(self, text: str) -> Dict[str, object]:
        """Parse a free-form user message into a mapping of variables.

        Extracts ``key=value`` or ``key:value`` pairs anywhere in the text,
        allowing optional whitespace and sign/punctuation in numeric values.
        Only keys present in the input mapping (or their mapped model names)
        are retained to prevent accidental passthrough of unrelated words.
        """

        pattern = re.compile(r"(?P<key>[\w_]+)\s*[:=]\s*(?P<value>[+-]?[\w\.]+)")
        parsed: Dict[str, object] = {}

        known_keys = set(self.input_mapping.keys()) | set(self.input_mapping.values())
        for match in pattern.finditer(text):
            key = match.group("key")
            if key not in known_keys:
                continue

            value_str = match.group("value")
            try:
                value: object = float(value_str)
            except ValueError:
                value = value_str
            parsed[key] = value

        return parsed

    def rewrap_english(self, human_response: Dict[str, object]) -> str:
        """Build an English summary of the mapped response for the user."""

        if not human_response:
            return "No response variables were produced."

        parts = [f"{key} set to {value}" for key, value in human_response.items()]
        return "; ".join(parts)

    def communicate(
        self, signal: Dict[str, object], model_fn: Callable[[Dict[str, object]], Dict[str, object]]
    ) -> Dict[str, object]:  # pragma: no cover - thin wrapper
        """Execute a model function with mapped inputs and outputs."""

        encoded_signal = self.encode_input(signal)
        normalized_signal = self.normalize_signal(encoded_signal)
        model_response = model_fn(normalized_signal)
        return self.decode_output(model_response)

    def mediate(
        self, user_text: str, model_fn: Callable[[Dict[str, object]], Dict[str, object]]
    ) -> Dict[str, object]:
        """Full pipeline: parse, normalize, respond with variables, rewrap."""

        parsed_human = self.parse_user_signal(user_text)
        encoded_signal = self.encode_input(parsed_human)
        normalized_signal = self.normalize_signal(encoded_signal)
        model_response = model_fn(normalized_signal)
        human_response = self.decode_output(model_response)
        english_summary = self.rewrap_english(human_response)

        return {
            "input_variables": normalized_signal,
            "model_response": model_response,
            "human_response": human_response,
            "english_response": english_summary,
        }


@dataclass
class ChatGPTClient:
    """Minimal wrapper to call the OpenAI Chat Completions API.

    Designed to keep payloads small: defaults to a compact model, low
    ``max_tokens``, and short prompts. The client can be stress-tested by
    repeated invocations using ``stress_test_token_usage``.
    """

    api_key: str | None = None
    model: str = "gpt-4o-mini"
    endpoint: str = "https://api.openai.com/v1/chat/completions"

    def __post_init__(self) -> None:
        key = self.api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("An OpenAI API key is required")
        object.__setattr__(self, "api_key", key)

    def send(
        self,
        messages: List[Dict[str, str]],
        *,
        max_tokens: int = 16,
        temperature: float = 0.0,
        **extra: object,
    ) -> Dict[str, object]:
        """Send a chat completion request with a compact payload."""

        payload: Dict[str, object] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        payload.update(extra)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:  # pragma: no cover - passthrough
            raise RuntimeError(f"ChatGPT request failed: {exc.read()}") from exc

        return json.loads(body)


@dataclass
class WrapperTerminal:
    """Interactive terminal that sits in front of the communication wrapper.

    The terminal prompts the user for an API key before entering a wrapper loop
    that accepts free-form text, normalizes the variables, and sends a minimal
    ChatGPT request using the parsed mappings.
    """

    wrapper: CommunicationWrapper
    client_builder: Callable[[str], ChatGPTClient] = ChatGPTClient
    input_fn: Callable[[str], str] = input
    output_fn: Callable[[str], None] = print
    max_tokens: int = 8

    def prompt_api_key(self) -> ChatGPTClient:
        """Request an API key before any wrapper interaction occurs."""

        api_key = self.input_fn("Enter OpenAI API key (kept local): ").strip()
        if not api_key:
            raise ValueError("An API key is required to start the wrapper terminal")

        return self.client_builder(api_key=api_key)

    def _emit_response(
        self, prompt: List[Dict[str, str]], response: Dict[str, object], english: str
    ) -> None:
        split = extract_wrapped_outputs(response)
        summary = {
            "prompt": prompt,
            "english_response": english,
            "chatgpt_response": response,
        }
        if split:
            summary["wrapped_outputs"] = split
        self.output_fn(json.dumps(summary, indent=2))

    def prompt_loop(self, client: ChatGPTClient) -> None:
        """Enter an embedded wrapper terminal after collecting the API key."""

        self.output_fn(
            "Enter wrapper commands (type 'exit' or 'quit' to leave).\n"
            "Provide key=value pairs, e.g., speed=4 theta:-45"
        )

        while True:
            user_text = self.input_fn("wrapper> ").strip()
            if user_text.lower() in {"exit", "quit"}:
                self.output_fn("Exiting wrapper terminal.")
                break
            if not user_text:
                continue

            prompt = build_minimal_prompt(self.wrapper, user_text)
            mediated = self.wrapper.mediate(user_text, lambda payload: payload)

            try:
                response = client.send(prompt, max_tokens=self.max_tokens)
            except Exception as exc:  # pragma: no cover - passthrough for operators
                self.output_fn(f"Error contacting ChatGPT: {exc}")
                continue

            self._emit_response(prompt, response, mediated["english_response"])


def build_minimal_prompt(
    wrapper: CommunicationWrapper, user_text: str
) -> List[Dict[str, str]]:
    """Construct a terse prompt that echoes normalized variables.

    The prompt keeps the message footprint minimal while communicating the
    normalized variable map to the model. It also requests that the model
    operate in a 3D scene description and return **two** separate pieces of
    output per turn:

    1. ``text_response`` — conversational wording based solely on the mapped
       variables (no raw user text).
    2. ``image`` — a single generated or retrieved visual (e.g., URL or data
       URI) describing the same variables in 3D space.

    The system message explicitly instructs the model to ignore raw user text
    and rely only on the normalized variables.
    """

    parsed = wrapper.parse_user_signal(user_text)
    encoded = wrapper.encode_input(parsed)
    normalized = wrapper.normalize_signal(encoded)

    system_prompt = (
        "You work in a 3D scene. Use ONLY the provided normalized variables "
        "(no user text) to craft JSON with two keys: 'text_response' for a "
        "concise conversational reply, and 'image' containing either a "
        "generated or retrieved visual (url/data + short alt)."
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"vars={normalized}"},
    ]


def extract_wrapped_outputs(response: Dict[str, object]) -> Dict[str, object]:
    """Attempt to split text and image responses from a ChatGPT payload."""

    try:
        choices = response.get("choices")  # type: ignore[assignment]
        if not choices:
            return {}

        message = choices[0].get("message", {})  # type: ignore[index]
        content = message.get("content", "")
        payload = json.loads(content)
    except Exception:
        return {}

    text_response = payload.get("text_response")
    image = payload.get("image")

    result: Dict[str, object] = {}
    if text_response is not None:
        result["text_response"] = text_response
    if image is not None:
        result["image"] = image
    return result


def stress_test_token_usage(
    client: ChatGPTClient,
    wrapper: CommunicationWrapper,
    user_text: str,
    *,
    runs: int = 3,
    max_tokens: int = 8,
) -> Dict[str, object]:
    """Execute repeated minimal calls to track usage footprints.

    Returns the shared prompt and each raw response so callers can inspect
    token usage without inflating the number of generated tokens.
    """

    prompt = build_minimal_prompt(wrapper, user_text)
    responses: List[Dict[str, object]] = []
    for _ in range(runs):
        responses.append(
            client.send(prompt, max_tokens=max_tokens, temperature=0.0)
        )

    return {"prompt": prompt, "responses": responses}


def launch_wrapper_terminal() -> None:
    """Run a simple executable wrapper with API key gating and terminal loop."""

    default_wrapper = CommunicationWrapper(
        input_mapping={"speed": "v", "angle": "theta", "height": "z"},
        output_mapping={"v": "speed", "theta": "angle", "z": "height"},
        variable_ranges={"v": (0.0, 10.0), "theta": (-180.0, 180.0), "z": (0.0, 20.0)},
    )

    terminal = WrapperTerminal(default_wrapper, input_fn=lambda prompt: (
        getpass(prompt) if "API key" in prompt else input(prompt)
    ))

    try:
        client = terminal.prompt_api_key()
    except ValueError as exc:  # pragma: no cover - user-facing
        print(exc)
        sys.exit(1)

    terminal.prompt_loop(client)


if __name__ == "__main__":  # pragma: no cover - manual execution only
    launch_wrapper_terminal()


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
