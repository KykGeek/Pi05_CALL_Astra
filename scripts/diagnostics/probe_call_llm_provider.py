"""No-robot diagnostic for the Codex app-server provider and tool-call path."""
from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import struct
import subprocess
import tempfile
import zlib
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from call_llm.runtime.codex_client import CodexAppServerClient, CodexClientError


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("text", "tool", "exec"), default="tool")
    parser.add_argument("--provider-variant", choices=("custom", "openai"), default="custom")
    parser.add_argument("--model", default="gpt-6-luna")
    parser.add_argument("--effort", choices=("low", "medium", "high", "xhigh", "max"), default="medium")
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--relay", action="store_true",
                        help="Route the no-robot Codex probe through a loopback HTTPS relay")
    parser.add_argument("--with-images", action="store_true",
                        help="Exercise two synthetic RGB attachments in the no-robot tool loop")
    args = parser.parse_args()
    base_url = os.environ.get("BASE_URL", "https://api.zhizengzeng.com/v1").rstrip("/")

    if args.mode == "exec":
        executable = shutil.which("codex")
        if executable is None:
            print(json.dumps({"probe": "failed", "mode": "exec", "error_code": "codex_cli_not_found"}))
            return 1
        provider_id = "zhizengzeng" if args.provider_variant == "custom" else "openai"
        auth_env = "API_SECRET_KEY" if provider_id == "zhizengzeng" else "OPENAI_API_KEY"
        config = [
            'model_provider="' + provider_id + '"',
            "model_providers." + provider_id + ".name=" + json.dumps(provider_id),
            "model_providers." + provider_id + ".base_url=" + json.dumps(base_url),
            "model_providers." + provider_id + '.wire_api="responses"',
            "model_providers." + provider_id + ".env_key=" + json.dumps(auth_env),
            "model_providers." + provider_id + ".requires_openai_auth=" + (
                "false" if provider_id == "zhizengzeng" else "true"
            ),
            'model_reasoning_effort="' + args.effort + '"',
            "features.multi_agent_v2.enabled=false",
        ]
        with tempfile.TemporaryDirectory(prefix="astra_codex_exec_probe_") as workspace:
            command = [
                executable, "exec", "--ignore-user-config", "--ephemeral",
                "--skip-git-repo-check", "--sandbox", "read-only", "--cd", workspace,
            ]
            for value in config:
                command.extend(("-c", value))
            command.extend(("--model", args.model, "Reply with the single word OK."))
            child_env = os.environ.copy()
            if provider_id == "openai":
                child_env["OPENAI_API_KEY"] = child_env.get("API_SECRET_KEY", "")
            try:
                completed = subprocess.run(
                    command, cwd=workspace, env=child_env, text=True,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    timeout=float(args.timeout), check=False,
                )
                result = {
                    "probe": "passed" if completed.returncode == 0 else "failed",
                    "mode": "exec",
                    "provider_variant": args.provider_variant,
                    "exit_code": int(completed.returncode),
                    "ok_response_present": "OK" in completed.stdout,
                    "response_stream_disconnected": (
                        "responsestreamdisconnected" in completed.stderr.lower()
                        or "stream disconnected" in completed.stderr.lower()
                    ),
                }
            except subprocess.TimeoutExpired:
                result = {"probe": "failed", "mode": "exec", "error_code": "codex_exec_timeout"}
        print(json.dumps(result, sort_keys=True))
        return 0 if result["probe"] == "passed" else 1

    ping_tool = {
        "type": "function",
        "name": "diagnostic_ping",
        "description": "Return the supplied boolean. This is a no-op diagnostic tool.",
        "inputSchema": {
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
            "additionalProperties": False,
        },
    }
    if args.with_images and args.mode != "tool":
        parser.error("--with-images is available only with --mode tool")
    observe_tool = {
        "type": "function",
        "name": "diagnostic_observe",
        "description": "Return a synthetic no-robot observation containing two RGB images.",
        "inputSchema": {"type": "object", "properties": {}, "required": [],
                        "additionalProperties": False},
    }
    dynamic_tools = ([observe_tool, ping_tool] if args.mode == "tool" and args.with_images
                     else [ping_tool] if args.mode == "tool" else [])
    if args.mode == "tool" and args.with_images:
        developer_instructions = (
            "This is a no-robot transport diagnostic. Call diagnostic_observe exactly once, "
            "inspect both returned synthetic images, then call diagnostic_ping exactly once "
            "with ok=true. Never claim these are real robot images."
        )
        prompt = "Call diagnostic_observe once, then diagnostic_ping once with ok=true."
    else:
        developer_instructions = (
            "For this protocol probe, call diagnostic_ping exactly once with ok=true."
            if args.mode == "tool"
            else "For this protocol probe, answer with the single word OK."
        )
        prompt = (
            "Call diagnostic_ping exactly once with ok=true."
            if args.mode == "tool"
            else "Reply with the single word OK."
        )
    result = {"probe": "failed"}
    with tempfile.TemporaryDirectory(prefix="astra_codex_probe_") as workspace:
        client = CodexAppServerClient(
            workspace=workspace,
            model=args.model,
            effort=args.effort,
            developer_instructions=developer_instructions,
            dynamic_tools=dynamic_tools,
            max_wall_seconds=float(args.timeout),
            use_provider_relay=args.relay,
        )
        try:
            client.start()
            def dispatch(name, values):
                if name == "diagnostic_observe" and args.with_images:
                    return {
                        "observation": "Synthetic RGB only; no robot, task, or simulator is connected.",
                        "_images": [
                            {"label": "synthetic RGB view A", "data_url": _png_data_url((220, 70, 55))},
                            {"label": "synthetic RGB view B", "data_url": _png_data_url((45, 110, 210))},
                        ],
                        "_terminal": False,
                    }
                if name == "diagnostic_ping":
                    return {"ok": bool(values.get("ok")), "_terminal": True}
                return {"ok": False, "_terminal": False}

            terminal = client.run(
                prompt, dispatch,
                idle_timeout_seconds=float(args.timeout),
                max_turns_without_terminal_tool=0,
            )
            result = {"probe": "passed", "mode": args.mode}
            if args.mode == "tool":
                result.update({
                    "terminal_tool_received": bool(terminal.get("terminal")),
                    "tool_calls": int(terminal.get("tool_calls", 0)),
                })
            result["cli_version_present"] = client.cli_version != "unknown"
        except CodexClientError as error:
            if args.mode == "text" and error.code == "model_completed_without_terminal_tool":
                result = {
                    "probe": "passed",
                    "mode": "text",
                    "turn_completed": True,
                    "cli_version_present": client.cli_version != "unknown",
                }
            else:
                result = {
                    "probe": "failed",
                    "mode": args.mode,
                    "error_code": error.code,
                    "model_response_received": bool(error.model_response_received),
                    "cli_version_present": client.cli_version != "unknown",
                }
        finally:
            if args.relay:
                result["relay"] = client.provider_transport_summary
            client.close()
    print(json.dumps(result, sort_keys=True))
    return 0 if result.get("probe") == "passed" else 1


def _png_data_url(color):
    """Create a tiny valid synthetic PNG in memory; no diagnostic files persist."""
    width = height = 48
    red, green, blue = color
    rows = []
    for y in range(height):
        row = bytearray([0])
        for x in range(width):
            if 8 <= x < 40 and 8 <= y < 40:
                row.extend((red, green, blue))
            else:
                row.extend((245, 245, 245))
        rows.append(bytes(row))

    def chunk(kind, payload):
        return (struct.pack(">I", len(payload)) + kind + payload +
                struct.pack(">I", zlib.crc32(kind + payload) & 0xffffffff))

    image = (b"\x89PNG\r\n\x1a\n" +
             chunk(b"IHDR", struct.pack(">2I5B", width, height, 8, 2, 0, 0, 0)) +
             chunk(b"IDAT", zlib.compress(b"".join(rows))) + chunk(b"IEND", b""))
    return "data:image/png;base64," + base64.b64encode(image).decode("ascii")


if __name__ == "__main__":
    raise SystemExit(main())
