"""Test SGLang's Chat Completions, Responses, and Anthropic Messages APIs with a forced tool call.

    uv run src/test_endpoints.py [BASE_URL]

BASE_URL may include /v1. When omitted, the script discovers the running Runpod pod named in
config.yaml. A test passes only when the endpoint returns a structured call to get_weather.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import requests
import runpod
import yaml
from dotenv import load_dotenv

TOOL_NAME = "get_weather"
TOOL_DESCRIPTION = "Get the current weather for a city."
TOOL_PARAMETERS = {
    "type": "object",
    "properties": {"city": {"type": "string", "description": "City name"}},
    "required": ["city"],
    "additionalProperties": False,
}
PROMPT = "Use the get_weather tool to check the weather in Paris. Do not answer from memory."


def post_json(url: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
    response = requests.post(url, headers=headers, json=payload, timeout=180)
    if not response.ok:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text[:2000]}")
    return response.json()


def parse_arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        return json.loads(value)
    raise AssertionError(f"unexpected tool arguments: {value!r}")


def verify_call(name: Any, arguments: Any) -> str:
    if name != TOOL_NAME:
        raise AssertionError(f"expected {TOOL_NAME!r}, got {name!r}")
    parsed = parse_arguments(arguments)
    if parsed.get("city", "").lower() != "paris":
        raise AssertionError(f"expected city='Paris', got {parsed!r}")
    return f"{name}({json.dumps(parsed, sort_keys=True)})"


def test_chat_completions(root: str, model: str, headers: dict[str, str]) -> str:
    data = post_json(
        f"{root}/v1/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": PROMPT}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": TOOL_NAME,
                        "description": TOOL_DESCRIPTION,
                        "parameters": TOOL_PARAMETERS,
                    },
                }
            ],
            "tool_choice": {"type": "function", "function": {"name": TOOL_NAME}},
            "temperature": 0,
            "max_tokens": 128,
        },
        headers,
    )
    calls = data["choices"][0]["message"].get("tool_calls") or []
    if not calls:
        raise AssertionError(f"no tool_calls in response: {json.dumps(data)[:2000]}")
    function = calls[0]["function"]
    return verify_call(function["name"], function["arguments"])


def test_responses(root: str, model: str, headers: dict[str, str]) -> str:
    data = post_json(
        f"{root}/v1/responses",
        {
            "model": model,
            "input": PROMPT,
            "tools": [
                {
                    "type": "function",
                    "name": TOOL_NAME,
                    "description": TOOL_DESCRIPTION,
                    "parameters": TOOL_PARAMETERS,
                    "strict": True,
                }
            ],
            "tool_choice": {"type": "function", "name": TOOL_NAME},
            "temperature": 0,
            "max_output_tokens": 128,
            "store": False,
        },
        headers,
    )
    calls = [item for item in data.get("output", []) if item.get("type") == "function_call"]
    if not calls:
        raise AssertionError(f"no function_call in response output: {json.dumps(data)[:2000]}")
    return verify_call(calls[0]["name"], calls[0]["arguments"])


def test_anthropic_messages(root: str, model: str, headers: dict[str, str]) -> str:
    anthropic_headers = {**headers, "anthropic-version": "2023-06-01"}
    data = post_json(
        f"{root}/v1/messages",
        {
            "model": model,
            "messages": [{"role": "user", "content": PROMPT}],
            "tools": [
                {
                    "name": TOOL_NAME,
                    "description": TOOL_DESCRIPTION,
                    "input_schema": TOOL_PARAMETERS,
                }
            ],
            "tool_choice": {"type": "tool", "name": TOOL_NAME},
            "temperature": 0,
            "max_tokens": 128,
        },
        anthropic_headers,
    )
    calls = [item for item in data.get("content", []) if item.get("type") == "tool_use"]
    if not calls:
        raise AssertionError(f"no tool_use in response content: {json.dumps(data)[:2000]}")
    return verify_call(calls[0]["name"], calls[0]["input"])


def discover_base_url(config: dict[str, Any]) -> str:
    runpod.api_key = os.environ["RUNPOD_API_KEY"]
    name = config["pod"]["name"]
    pod = next(
        (item for item in runpod.get_pods() if item["name"] == name and item["desiredStatus"] == "RUNNING"),
        None,
    )
    if pod is None:
        raise SystemExit(f"No running Runpod pod named {name!r}; pass BASE_URL explicitly or start it first.")
    return f"https://{pod['id']}-{config['server']['port']}.proxy.runpod.net"


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_url", nargs="?", help="SGLang API root, with or without /v1")
    parser.add_argument("--config", default="config.yaml", help="serving config used for model and pod discovery")
    args = parser.parse_args()

    config: dict[str, Any] = yaml.safe_load(Path(args.config).read_text())
    root = (args.base_url or os.environ.get("BASE_URL") or discover_base_url(config)).rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]

    api_key = os.environ.get("API_KEY")
    if not api_key:
        raise SystemExit("API_KEY must be set in .env")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    model = config["server"]["args"]["served-model-name"]

    tests = [
        ("Chat Completions", test_chat_completions),
        ("Responses", test_responses),
        ("Anthropic Messages", test_anthropic_messages),
    ]
    failures = 0
    print(f"Testing {model} at {root}")
    for name, test in tests:
        try:
            call = test(root, model, headers)
            print(f"PASS  {name:20} {call}")
        except Exception as error:
            failures += 1
            print(f"FAIL  {name:20} {error}")

    if failures:
        raise SystemExit(f"{failures}/{len(tests)} endpoint tests failed")
    print(f"All {len(tests)} endpoint tool-call tests passed.")


if __name__ == "__main__":
    main()
