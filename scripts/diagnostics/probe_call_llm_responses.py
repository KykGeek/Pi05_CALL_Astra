"""No-robot check of the provider's streamed Responses/tool schema path."""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
import sys

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from call_llm.runtime.executor import _developer_instructions
from call_llm.runtime.protocol import tool_specs


def _safe(value):
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.:-]{1,64}", value):
        return value
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt-6-luna")
    parser.add_argument("--effort", choices=("low", "medium", "high", "xhigh", "max"), default="medium")
    parser.add_argument("--timeout", type=float, default=90.0)
    args = parser.parse_args()
    api_key = os.environ.get("API_SECRET_KEY", "").strip()
    base_url = os.environ.get("BASE_URL", "https://api.zhizengzeng.com/v1").rstrip("/")
    if not api_key:
        raise SystemExit("API_SECRET_KEY is required")

    tools = [
        {
            "type": "function",
            "name": spec["name"],
            "description": spec["description"],
            "parameters": spec["inputSchema"],
        }
        for spec in tool_specs()
    ]
    body = {
        "model": args.model,
        "input": [
            {"role": "developer", "content": _developer_instructions()},
            {"role": "user", "content": (
                "This is a no-robot wire probe. Use episode_id=diagnostic-episode and "
                "intervention_id=diagnostic-intervention. Call libero_observe exactly once."
            )},
        ],
        "reasoning": {"effort": args.effort},
        "tools": tools,
        "tool_choice": "auto",
        "parallel_tool_calls": False,
        "max_output_tokens": 1024,
        "stream": True,
    }
    result = {
        "http_status": None,
        "content_type": None,
        "event_count": 0,
        "event_types": [],
        "tool_names": [],
        "completed_event": False,
        "error_type": None,
        "error_code": None,
    }
    try:
        response = requests.post(
            base_url + "/responses",
            headers={"Authorization": "Bearer " + api_key,
                     "Content-Type": "application/json"},
            json=body,
            stream=True,
            timeout=(15, float(args.timeout)),
        )
        result["http_status"] = response.status_code
        result["content_type"] = response.headers.get("content-type", "").split(";")[0]
        if response.status_code >= 400:
            try:
                error = response.json().get("error", {})
                if isinstance(error, dict):
                    result["error_type"] = _safe(error.get("type"))
                    result["error_code"] = _safe(error.get("code"))
            except (ValueError, AttributeError):
                pass
        else:
            for raw in response.iter_lines(decode_unicode=True):
                if not raw:
                    continue
                line = raw if isinstance(raw, str) else raw.decode("utf-8", "replace")
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    continue
                try:
                    event = json.loads(data)
                except ValueError:
                    continue
                kind = event.get("type")
                if isinstance(kind, str):
                    result["event_count"] += 1
                    if kind not in result["event_types"]:
                        result["event_types"].append(kind)
                    result["completed_event"] |= kind == "response.completed"
                item = event.get("item")
                if isinstance(item, dict) and item.get("type") == "function_call":
                    name = item.get("name")
                    if isinstance(name, str) and name not in result["tool_names"]:
                        result["tool_names"].append(name)
                if kind == "response.output_item.added":
                    output_item = event.get("item")
                    if isinstance(output_item, dict) and output_item.get("type") == "function_call":
                        name = output_item.get("name")
                        if isinstance(name, str) and name not in result["tool_names"]:
                            result["tool_names"].append(name)
    except requests.RequestException as error:
        result["transport_error_type"] = type(error).__name__
    print(json.dumps(result, sort_keys=True))
    return 0 if result["http_status"] == 200 and result["completed_event"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
