"""Unit tests for the CommunicationWrapper mediation flow."""

from unittest.mock import MagicMock, patch

from rc_trajectory import (
    ChatGPTClient,
    CommunicationWrapper,
    WrapperTerminal,
    build_minimal_prompt,
    extract_wrapped_outputs,
    stress_test_token_usage,
)


def make_wrapper():
    return CommunicationWrapper(
        input_mapping={"speed": "v", "angle": "theta", "height": "z"},
        output_mapping={"v": "speed", "theta": "angle", "z": "height"},
        variable_ranges={"v": (0.0, 10.0), "theta": (-180.0, 180.0), "z": (0.0, 20.0)},
    )


def test_parse_user_signal_supports_embedded_pairs():
    wrapper = make_wrapper()

    parsed = wrapper.parse_user_signal("set speed=4.5 and theta:-90 now")

    assert parsed == {"speed": 4.5, "theta": -90.0}


def test_mediation_normalizes_and_rewraps_response():
    wrapper = make_wrapper()

    def model_fn(payload):
        # model receives normalized values and replies in model keys
        assert payload == {"v": 0.5, "theta": 0.625}
        return {"v": payload["v"] * 2, "theta": payload["theta"] / 2}

    result = wrapper.mediate("speed=5, theta:45", model_fn)

    assert result["input_variables"] == {"v": 0.5, "theta": 0.625}
    assert result["model_response"] == {"v": 1.0, "theta": 0.3125}
    assert result["human_response"] == {"speed": 1.0, "angle": 0.3125}
    assert result["english_response"] == "speed set to 1.0; angle set to 0.3125"


def test_build_minimal_prompt_is_terse():
    wrapper = make_wrapper()

    prompt = build_minimal_prompt(wrapper, "speed=5 theta=90")

    assert prompt[0]["role"] == "system"
    assert "3D scene" in prompt[0]["content"]
    assert "text_response" in prompt[0]["content"]
    assert "image" in prompt[0]["content"]
    assert prompt[1]["content"] == "vars={'v': 0.5, 'theta': 0.75}"


def test_build_minimal_prompt_handles_height_dimension():
    wrapper = make_wrapper()

    prompt = build_minimal_prompt(wrapper, "speed=10 height=10")

    assert "'z': 0.5" in prompt[1]["content"]


def test_extract_wrapped_outputs_splits_text_and_image():
    payload = {
        "choices": [
            {
                "message": {
                    "content": "{\"text_response\": \"ok\", \"image\": {\"url\": \"http://example.com/img.png\", \"alt\": \"a scene\"}}"
                }
            }
        ]
    }

    result = extract_wrapped_outputs(payload)

    assert result["text_response"] == "ok"
    assert result["image"]["url"].endswith("img.png")


@patch("rc_trajectory.urllib.request.urlopen")
def test_chatgpt_client_sends_compact_payload(mock_urlopen):
    wrapper = make_wrapper()

    mock_response = MagicMock()
    mock_response.read.return_value = b"{\"choices\": [], \"usage\": {\"total_tokens\": 7}}"
    mock_urlopen.return_value.__enter__.return_value = mock_response

    client = ChatGPTClient(api_key="test-key")
    prompt = build_minimal_prompt(wrapper, "speed=3")
    result = client.send(prompt, max_tokens=4)

    assert result["usage"]["total_tokens"] == 7
    mock_urlopen.assert_called_once()
    request_obj = mock_urlopen.call_args.args[0]
    assert request_obj.headers["Authorization"] == "Bearer test-key"
    payload = request_obj.data.decode("utf-8")
    assert "\"max_tokens\": 4" in payload


def test_stress_test_token_usage_repeats_requests():
    wrapper = make_wrapper()
    client = MagicMock()
    client.send.side_effect = [
        {"usage": {"prompt_tokens": 5}},
        {"usage": {"prompt_tokens": 5}},
    ]

    result = stress_test_token_usage(
        client, wrapper, "speed=2", runs=2, max_tokens=6
    )

    assert len(result["responses"]) == 2
    assert all(r["usage"]["prompt_tokens"] == 5 for r in result["responses"])
    client.send.assert_called_with(
        result["prompt"], max_tokens=6, temperature=0.0
    )


def test_wrapper_terminal_prompts_for_key_then_runs_loop(monkeypatch):
    wrapper = make_wrapper()
    inputs = iter(["test-key", "speed=4 theta:90", "quit"])
    captured: list[str] = []

    def fake_input(prompt: str) -> str:
        return next(inputs)

    def fake_output(message: str) -> None:
        captured.append(message)

    client = MagicMock()
    client.send.return_value = {
        "choices": [
            {
                "message": {
                    "content": "{\"text_response\": \"hi there\", \"image\": {\"url\": \"http://example.com/x.png\"}}"
                }
            }
        ]
    }
    builder = MagicMock(return_value=client)

    terminal = WrapperTerminal(
        wrapper,
        client_builder=builder,
        input_fn=fake_input,
        output_fn=fake_output,
        max_tokens=4,
    )

    built_client = terminal.prompt_api_key()

    assert built_client is client
    builder.assert_called_once_with(api_key="test-key")

    terminal.prompt_loop(client)

    assert any("english_response" in message for message in captured)
    assert any("wrapped_outputs" in message for message in captured)
    client.send.assert_called_once()
