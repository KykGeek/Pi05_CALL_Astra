"""Bounded JSON-RPC client for episode-scoped dynamic tools."""
from __future__ import annotations

import json
import ipaddress
import os
from pathlib import Path
import queue
import re
import shutil
import signal
import subprocess
import threading
import time
from typing import Any, Callable, Dict, Mapping, Optional
from urllib.parse import urlsplit

from .provider_proxy import LocalResponsesProxy


DEFAULT_MODEL_PROVIDER = os.environ.get("ASTRA_CODEX_PROVIDER", "custom")
DEFAULT_BASE_URL = os.environ.get("ASTRA_BASE_URL", "")
API_KEY_ENVIRONMENT_VARIABLE = os.environ.get("ASTRA_API_KEY_ENV", "ASTRA_API_KEY")
DEFAULT_ASTRA_MODEL = os.environ.get("ASTRA_MODEL", "gpt-6-astra")
DEFAULT_REASONING_EFFORT = os.environ.get("ASTRA_REASONING_EFFORT", "medium")


class CodexClientError(RuntimeError):
    def __init__(self, code: str, *, model_response_received: bool = False) -> None:
        super().__init__(code)
        self.code = code
        self.model_response_received = bool(model_response_received)


class CodexAppServerClient:
    """Small JSON-RPC client; only the host's dynamic LIBERO tools can act."""

    def __init__(self, *, workspace: str, model: str, effort: str,
                 developer_instructions: str, dynamic_tools: list[dict[str, Any]],
                 max_wall_seconds: float = 300.0, base_url: Optional[str] = None,
                 allow_loopback_http: bool = False,
                 use_provider_relay: bool = False) -> None:
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.model = str(model)
        self.effort = str(effort)
        self.model_provider = os.environ.get("ASTRA_CODEX_PROVIDER", DEFAULT_MODEL_PROVIDER)
        self.base_url = (base_url or os.environ.get("ASTRA_BASE_URL", DEFAULT_BASE_URL)).rstrip("/")
        self.allow_loopback_http = bool(allow_loopback_http)
        self.use_provider_relay = bool(use_provider_relay)
        self.api_key_environment_variable = os.environ.get(
            "ASTRA_API_KEY_ENV", API_KEY_ENVIRONMENT_VARIABLE
        ).strip()
        self.developer_instructions = str(developer_instructions)
        self.dynamic_tools = dynamic_tools
        self.max_wall_seconds = float(max_wall_seconds)
        if self.max_wall_seconds <= 0:
            raise ValueError("invalid_Astra_wall_budget")
        self._incoming: "queue.Queue[Any]" = queue.Queue()
        self._responses: Dict[Any, "queue.Queue[Any]"] = {}
        self._lock = threading.RLock()
        self._next_id = 1
        self._proc: Optional[subprocess.Popen[str]] = None
        self._provider_proxy: Optional[LocalResponsesProxy] = None
        self._last_provider_proxy_summary: Optional[dict[str, Any]] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._stdout_thread: Optional[threading.Thread] = None
        self._model_response_received = False
        self.cli_version = "unknown"
        self.thread_id: Optional[str] = None

    @property
    def model_response_received(self) -> bool:
        return self._model_response_received

    def start(self) -> None:
        if self._proc is not None:
            raise RuntimeError("Codex_client_already_started")
        if not self.api_key_environment_variable or not os.environ.get(
            self.api_key_environment_variable, ""
        ).strip():
            raise CodexClientError("astra_provider_api_key_missing")
        if not self.base_url:
            raise CodexClientError("astra_provider_base_url_missing")
        if not _valid_provider_url(self.base_url, allow_loopback_http=self.allow_loopback_http):
            raise CodexClientError("astra_provider_base_url_invalid")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", self.model_provider):
            raise CodexClientError("astra_provider_name_invalid")
        executable = shutil.which("codex")
        if not executable:
            raise CodexClientError("codex_cli_not_found")
        profile = "astra_libero_episode"
        codex_base_url = self.base_url
        if self.use_provider_relay:
            try:
                self._provider_proxy = LocalResponsesProxy(
                    self.base_url,
                    allowed_tool_names=[
                        str(tool.get("name")) for tool in self.dynamic_tools
                        if isinstance(tool, Mapping) and isinstance(tool.get("name"), str)
                    ],
                )
            except ValueError as error:
                raise CodexClientError(str(error)) from error
            codex_base_url = self._provider_proxy.base_url
        argv = self._build_argv(executable, profile, base_url=codex_base_url)
        kwargs: Dict[str, Any] = {
            "cwd": str(self.workspace),
            "stdin": subprocess.PIPE,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "text": True,
            "bufsize": 1,
        }
        if os.name == "nt":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        else:
            kwargs["start_new_session"] = True
        try:
            self._proc = subprocess.Popen(argv, **kwargs)
        except Exception as error:
            raise CodexClientError("codex_app_server_start_failed") from error
        self._stdout_thread = threading.Thread(target=self._read_stdout, daemon=True)
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stdout_thread.start()
        self._stderr_thread.start()
        try:
            self.cli_version = _read_cli_version(executable)
            initialize = self._request("initialize", {
                "clientInfo": {"name": "libero-astra-recovery", "version": "1.0"},
                "capabilities": {"experimentalApi": True},
            }, timeout=30.0)
            server_version = str(initialize.get("serverInfo", {}).get("version", "unknown"))
            if self.cli_version == "unknown" and re.fullmatch(r"[A-Za-z0-9_.+-]{1,64}", server_version):
                self.cli_version = server_version
            self._notify("initialized", {})
            thread = self._request("thread/start", {
                "cwd": str(self.workspace),
                "model": self.model,
                "modelProvider": self.model_provider,
                "config": {"model_reasoning_effort": self.effort},
                "developerInstructions": self.developer_instructions,
                "dynamicTools": self.dynamic_tools,
                "ephemeral": True,
                "allowProviderModelFallback": False,
                "approvalPolicy": "never",
                "permissions": profile,
                "runtimeWorkspaceRoots": [str(self.workspace)],
            }, timeout=45.0)
            actual_model = str(thread.get("model", ""))
            actual_provider = str(thread.get("modelProvider", ""))
            actual_effort = str(thread.get("reasoningEffort", ""))
            if (actual_model != self.model or actual_provider != self.model_provider or
                    actual_effort != self.effort):
                raise CodexClientError("codex_model_or_reasoning_effort_mismatch")
            self.thread_id = str(thread["thread"]["id"])
        except Exception as error:
            self.close()
            if isinstance(error, CodexClientError):
                raise
            raise CodexClientError("codex_thread_initialization_failed") from error

    def _build_argv(self, executable: str, profile: str,
                    *, base_url: Optional[str] = None) -> list[str]:
        filesystem = {
            ":root": "deny",
            ":minimal": "read",
            str(self.workspace): "read",
        }
        filesystem_override = "{" + ", ".join(
            json.dumps(path) + " = " + json.dumps(access)
            for path, access in filesystem.items()
        ) + "}"
        argv = [
            executable, "app-server", "--stdio", "--strict-config",
            "-c", 'default_permissions="' + profile + '"',
            # Disable subagents to keep each episode isolated.
            "-c", "features.multi_agent_v2.enabled=false",
            # The loopback relay sends only allowlisted dynamic tool schemas.
            "-c", "permissions." + profile + '.extends=":read-only"',
            "-c", "permissions." + profile + ".filesystem=" + filesystem_override,
            "-c", "permissions." + profile + ".network.enabled=false",
            "-c", "model_provider=" + json.dumps(self.model_provider),
            "-c", "model_providers." + self.model_provider + ".name=" + json.dumps(self.model_provider),
            "-c", "model_providers." + self.model_provider + ".base_url=" +
                  json.dumps(base_url or self.base_url),
            "-c", "model_providers." + self.model_provider + '.wire_api="responses"',
            "-c", "model_providers." + self.model_provider + ".supports_websockets=false",
            "-c", "model_providers." + self.model_provider + ".env_key=" +
                  json.dumps(self.api_key_environment_variable),
            "-c", "model_providers." + self.model_provider + ".requires_openai_auth=false",
            "-c", "model=" + json.dumps(self.model),
        ]
        return argv

    def run(self, initial_text: str, dispatch: Callable[[str, Mapping[str, Any]], Mapping[str, Any]],
            *, idle_timeout_seconds: float = 120.0,
            max_turns_without_terminal_tool: int = 1) -> Dict[str, Any]:
        if self._proc is None or self.thread_id is None:
            raise CodexClientError("codex_client_not_started")
        absolute_deadline = time.monotonic() + self.max_wall_seconds
        turns = 0
        tool_calls = 0
        self._start_turn(initial_text)
        turns += 1
        while time.monotonic() < absolute_deadline:
            timeout = min(float(idle_timeout_seconds), absolute_deadline - time.monotonic())
            try:
                event = self._next(timeout)
            except queue.Empty as error:
                raise CodexClientError("model_idle_timeout",
                                       model_response_received=self._model_response_received) from error
            method = event.get("method") if isinstance(event, Mapping) else None
            if method == "item/tool/call":
                self._model_response_received = True
                tool_calls += 1
                params = event.get("params") or {}
                name = params.get("tool")
                args = params.get("arguments") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except ValueError:
                        args = None
                if not isinstance(name, str) or not isinstance(args, Mapping):
                    result = {"success": False, "contentItems": [
                        {"type": "inputText", "text": '{"error":"invalid_tool_call","no_execution":true}'}
                    ]}
                    self._reply(event.get("id"), result)
                    continue
                try:
                    result_data = dispatch(name, args)
                    terminal = bool(result_data.get("_terminal", False))
                    packet = dict(result_data)
                    packet.pop("_terminal", None)
                    result = {"success": True, "contentItems": _content_items(packet)}
                except Exception as error:
                    code = getattr(error, "code", None) or getattr(error, "args", ["tool_rejected"])[0]
                    safe_code = _safe_code(code)
                    result = {"success": False, "contentItems": [
                        {"type": "inputText", "text": json.dumps({"error": safe_code, "no_execution": True})}
                    ]}
                    terminal = False
                self._reply(event.get("id"), result)
                if terminal:
                    return {"terminal": dict(result_data), "tool_calls": tool_calls,
                            "turns": turns, "model_response_received": True,
                            "cli_version": self.cli_version}
            elif method == "turn/completed":
                turns_completed = int(event.get("params", {}).get("turn", {}).get("id") is not None)
                if turns >= 1 + max_turns_without_terminal_tool:
                    raise CodexClientError("model_completed_without_terminal_tool",
                                           model_response_received=self._model_response_received)
                self._start_turn("Continue this same recovery. Use the latest tool observation, request and proposal IDs. Do not repeat completed actions. Make one next tool call.")
                turns += 1
            elif method in ("turn/failed", "error"):
                failure_code = _turn_failure_code(event) or _turn_failure_shape(event)
                suffix = "_" + failure_code if failure_code else ""
                raise CodexClientError("codex_turn_failed" + suffix,
                                       model_response_received=self._model_response_received)
        raise CodexClientError("intervention_wall_timeout",
                               model_response_received=self._model_response_received)

    def _start_turn(self, text: str) -> None:
        self._request("turn/start", {
            "threadId": self.thread_id,
            "model": self.model,
            "effort": self.effort,
            "approvalPolicy": "never",
            "cwd": str(self.workspace),
            "input": [{"type": "text", "text": str(text)}],
        }, timeout=45.0)

    def _request(self, method: str, params: Mapping[str, Any], timeout: float) -> Any:
        with self._lock:
            ident = self._next_id
            self._next_id += 1
            response_queue: "queue.Queue[Any]" = queue.Queue(maxsize=1)
            self._responses[ident] = response_queue
        try:
            proc = self._proc
            if proc is None or proc.stdin is None:
                raise CodexClientError("codex_process_closed")
            proc.stdin.write(json.dumps({"jsonrpc": "2.0", "id": ident,
                                         "method": method, "params": params}) + "\n")
            proc.stdin.flush()
            response = response_queue.get(timeout=timeout)
            if "error" in response:
                raise CodexClientError("codex_rpc_error_" + str(response["error"].get("code", "unknown")))
            return response.get("result") or {}
        except queue.Empty as error:
            raise CodexClientError("codex_rpc_timeout_" + _safe_code(method)) from error
        finally:
            with self._lock:
                self._responses.pop(ident, None)

    def _notify(self, method: str, params: Mapping[str, Any]) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None:
            raise CodexClientError("codex_process_closed")
        proc.stdin.write(json.dumps({"jsonrpc": "2.0", "method": method,
                                    "params": params}) + "\n")
        proc.stdin.flush()

    def _reply(self, ident: Any, result: Mapping[str, Any]) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None or ident is None:
            raise CodexClientError("codex_tool_reply_missing_id")
        proc.stdin.write(json.dumps({"jsonrpc": "2.0", "id": ident,
                                    "result": result}) + "\n")
        proc.stdin.flush()

    def _next(self, timeout: float) -> Mapping[str, Any]:
        message = self._incoming.get(timeout=max(0.0, timeout))
        if isinstance(message, BaseException):
            raise CodexClientError("codex_transport_closed",
                                   model_response_received=self._model_response_received)
        if not isinstance(message, Mapping):
            raise CodexClientError("invalid_codex_event")
        return message

    def _read_stdout(self) -> None:
        proc = self._proc
        stream = proc.stdout if proc is not None else None
        if stream is None:
            self._incoming.put(EOFError())
            return
        for line in stream:
            try:
                message = json.loads(line)
            except ValueError:
                # Startup diagnostics are intentionally not surfaced: they can
                # contain machine-local details unrelated to the robot trace.
                continue
            if not isinstance(message, Mapping):
                continue
            ident = message.get("id")
            if ident is not None and "method" not in message:
                with self._lock:
                    response_queue = self._responses.get(ident)
                if response_queue is not None:
                    response_queue.put(message)
                    continue
            self._incoming.put(message)
        self._incoming.put(EOFError())

    def _drain_stderr(self) -> None:
        proc = self._proc
        stream = proc.stderr if proc is not None else None
        if stream is not None:
            for _line in stream:
                pass

    def close(self) -> None:
        proc = self._proc
        self._proc = None
        try:
            if proc is not None and proc.stdin is not None:
                proc.stdin.close()
        except OSError:
            pass
        if proc is not None and proc.poll() is None:
            try:
                if os.name == "nt":
                    proc.terminate()
                else:
                    os.killpg(proc.pid, signal.SIGTERM)
            except OSError:
                try:
                    proc.terminate()
                except OSError:
                    pass
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    if os.name == "nt":
                        proc.kill()
                    else:
                        os.killpg(proc.pid, signal.SIGKILL)
                except OSError:
                    proc.kill()
        proxy = self._provider_proxy
        self._provider_proxy = None
        if proxy is not None:
            self._last_provider_proxy_summary = proxy.summary()
            proxy.close()

    @property
    def provider_transport_summary(self) -> Optional[dict[str, Any]]:
        if self._provider_proxy is not None:
            return self._provider_proxy.summary()
        return self._last_provider_proxy_summary


def _content_items(packet: Mapping[str, Any]) -> list[dict[str, Any]]:
    text = dict(packet)
    images = text.pop("_images", [])
    result = [{"type": "inputText", "text": json.dumps(text, ensure_ascii=False,
                                                        separators=(",", ":"), allow_nan=False)}]
    for image in images:
        if not isinstance(image, Mapping):
            continue
        label = str(image.get("label", "RGB observation"))
        data_url = image.get("data_url")
        if isinstance(data_url, str) and data_url.startswith("data:image/"):
            result.append({"type": "inputText", "text": label})
            result.append({"type": "inputImage", "imageUrl": data_url})
    return result


def _valid_provider_url(value: str, *, allow_loopback_http: bool = False) -> bool:
    parsed = urlsplit(value)
    common_invalid = (parsed.username or parsed.password or parsed.query or parsed.fragment or
                      parsed.path.rstrip("/") != "/v1")
    if common_invalid:
        return False
    if parsed.scheme == "https" and parsed.hostname:
        return parsed.port in (None, 443)
    if not allow_loopback_http or parsed.scheme != "http" or parsed.port is None:
        return False
    try:
        return ipaddress.ip_address(parsed.hostname or "").is_loopback
    except ValueError:
        return False


def _read_cli_version(executable: str) -> str:
    try:
        completed = subprocess.run(
            [executable, "--version"], capture_output=True, text=True,
            timeout=8.0, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    if completed.returncode != 0:
        return "unknown"
    match = re.search(r"\b(?:codex(?:-cli)?)\s+v?([0-9]+\.[0-9]+\.[0-9]+(?:[-+][A-Za-z0-9.-]+)?)\b",
                      completed.stdout.strip(), flags=re.IGNORECASE)
    return match.group(1) if match else "unknown"


def _safe_code(value: Any) -> str:
    text = str(value).strip().replace("\n", " ")[:120]
    if not text or any(token in text.lower() for token in ("key=", "token=", "password", "secret")):
        return "tool_rejected"
    return "".join(ch if ch.isalnum() or ch in "._:-" else "_" for ch in text)


def _turn_failure_code(event: Mapping[str, Any]) -> Optional[str]:
    """Extract only a short machine code from a failed turn.

    Provider messages and stderr may contain credentials or request details;
    they are deliberately discarded and never written to the episode audit.
    """
    params = event.get("params")
    candidates = [event.get("error")]
    if isinstance(params, Mapping):
        candidates.append(params.get("error"))
        turn = params.get("turn")
        if isinstance(turn, Mapping):
            candidates.append(turn.get("error"))
    # Some app-server/provider failures wrap the useful machine status one or
    # two levels below `error`. Traverse only
    # known error containers and only read a fixed set of machine-code fields;
    # never inspect or persist free-form messages.
    queue = [(item, 0) for item in candidates]
    while queue:
        item, depth = queue.pop(0)
        if isinstance(item, Mapping):
            for key in ("code", "error_code", "errorCode", "statusCode",
                        "status_code", "httpStatus", "http_status", "httpStatusCode",
                        "http_status_code", "type", "name"):
                value = item.get(key)
                if isinstance(value, int) and not isinstance(value, bool):
                    if 100 <= value <= 599 or -32768 <= value <= -1:
                        return str(value)
                if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.:-]{1,64}", value):
                    safe = _safe_code(value)
                    if safe != "tool_rejected":
                        return safe
            message = item.get("message")
            if isinstance(message, str):
                category = _safe_failure_category(message)
                if category is not None:
                    return category
            if depth < 4:
                for key in ("error", "cause", "codexErrorInfo", "providerError",
                            "provider_error", "responseStreamDisconnected",
                            "response_stream_disconnected", "details"):
                    child = item.get(key)
                    if isinstance(child, Mapping):
                        queue.append((child, depth + 1))
        elif isinstance(item, int) and not isinstance(item, bool) and 100 <= item <= 599:
            return str(item)
    return None


def _safe_failure_category(message: str) -> Optional[str]:
    """Classify provider failures without persisting their free-form message."""
    lowered = message.lower()
    status = re.search(r"\b(400|401|403|404|408|413|422|429|500|502|503|504)\b", lowered)
    if status:
        return {
            "400": "provider_request_rejected",
            "401": "provider_authentication_failed",
            "403": "provider_access_denied",
            "404": "provider_endpoint_or_model_not_found",
            "408": "provider_request_timeout",
            "413": "provider_request_too_large",
            "422": "provider_request_rejected",
            "429": "provider_rate_limited",
            "500": "provider_server_error",
            "502": "provider_gateway_error",
            "503": "provider_unavailable",
            "504": "provider_request_timeout",
        }[status.group(1)]
    if "invalid api key" in lowered or "incorrect api key" in lowered or "unauthorized" in lowered:
        return "provider_authentication_failed"
    if "rate limit" in lowered or "too many requests" in lowered:
        return "provider_rate_limited"
    if "model" in lowered and any(term in lowered for term in ("not found", "does not exist", "unknown")):
        return "provider_model_not_found"
    if any(term in lowered for term in ("unsupported", "not supported", "unknown endpoint")):
        return "provider_protocol_or_parameter_unsupported"
    if "timeout" in lowered or "timed out" in lowered:
        return "provider_request_timeout"
    return None


def _turn_failure_shape(event: Mapping[str, Any]) -> str:
    """Return field-presence diagnostics only; never copy provider values."""
    safe_fields = (
        "error", "turn", "status", "code", "error_code", "errorCode",
        "statusCode", "status_code", "httpStatus", "http_status", "type",
        "name", "message", "details", "codexErrorInfo", "providerError",
        "provider_error", "cause", "reason", "param", "request_id",
        "model_provider", "provider", "kind", "source", "responseStreamDisconnected",
        "response_stream_disconnected", "httpStatusCode", "http_status_code",
    )
    labels = []
    params = event.get("params")
    if isinstance(params, Mapping):
        labels.append("params")
        turn = params.get("turn")
        if isinstance(turn, Mapping):
            labels.append("turn")
            error = turn.get("error")
            if isinstance(error, Mapping):
                fields = [key for key in safe_fields if key in error]
                labels.append("turn_error_fields_" + ("_".join(fields) or "other"))
            elif error is not None:
                labels.append("turn_error_" + type(error).__name__.lower())
        error = params.get("error")
        if isinstance(error, Mapping):
            fields = [key for key in safe_fields if key in error]
            labels.append("params_error_fields_" + ("_".join(fields) or "other"))
            nested = error.get("codexErrorInfo")
            if isinstance(nested, Mapping):
                nested_fields = sorted(
                    str(key).lower()
                    for key in nested
                    if isinstance(key, str) and re.fullmatch(r"[A-Za-z0-9_.:-]{1,32}", key)
                )[:12]
                labels.append("codex_info_fields_" + ("_".join(nested_fields) or "other"))
    error = event.get("error")
    if isinstance(error, Mapping):
        fields = [key for key in safe_fields if key in error]
        labels.append("event_error_fields_" + ("_".join(fields) or "other"))
    elif error is not None:
        labels.append("event_error_" + type(error).__name__.lower())
    if not labels:
        labels.append("no_standard_error_fields")
    return "unclassified_" + "_".join(labels)[:180]
