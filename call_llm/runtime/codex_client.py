"""Bounded Codex app-server client for Astra's episode-scoped dynamic tools."""
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
import sys
import threading
import time
from typing import Any, Callable, Dict, Mapping, Optional
from urllib.parse import urlsplit

from .provider_proxy import LocalResponsesProxy


DEFAULT_MODEL_PROVIDER = "zhizengzeng"
DEFAULT_BASE_URL = "https://api.zhizengzeng.com/v1"
API_KEY_ENVIRONMENT_VARIABLE = "API_SECRET_KEY"
DEFAULT_ASTRA_MODEL = os.environ.get("ASTRA_MODEL", "gpt-6-luna")
DEFAULT_REASONING_EFFORT = os.environ.get("ASTRA_REASONING_EFFORT", "medium")


def _load_project_env() -> None:
    """Load the nearest project .env without overwriting explicit env vars."""
    roots: list[Path] = [Path(__file__).resolve().parents[2]]
    for candidate in (Path.cwd(),):
        roots.extend([candidate, *candidate.parents])
    seen: set[Path] = set()
    for root in roots:
        env_path = root / ".env"
        if env_path in seen or not env_path.is_file():
            continue
        seen.add(env_path)
        try:
            lines = env_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for raw in lines:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:].lstrip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            # Preserve an explicit non-empty process value, but allow the
            # project .env to repair variables exported as empty strings.
            if not key or (key in os.environ and os.environ[key].strip()):
                continue
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            os.environ[key] = value
        return


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
        self.base_url = (base_url or os.environ.get("BASE_URL", DEFAULT_BASE_URL)).rstrip("/")
        self.allow_loopback_http = bool(allow_loopback_http)
        self.use_provider_relay = bool(use_provider_relay)
        self.api_key_environment_variable = API_KEY_ENVIRONMENT_VARIABLE
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
        self._trace_enabled = os.environ.get("ASTRA_PROMPT_TRACE", "1").strip().lower() not in {
            "0", "false", "no", "off"
        }
        self._trace_path = self.workspace / "astra_prompt_trace.jsonl"
        self._trace_turn_index = 0
        self._trace_event_index = 0

    @property
    def model_response_received(self) -> bool:
        return self._model_response_received

    @staticmethod
    def _trace_estimated_tokens(value: Any) -> int:
        """Diagnostic estimate only; provider usage is recorded separately when available."""
        try:
            raw = json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
        except Exception:
            raw = str(value)
        return max(1, (len(raw) + 3) // 4)

    @staticmethod
    def _trace_sanitize_content_items(items: Any) -> list[Any]:
        """Keep text/state but omit camera base64 from the diagnostic trace."""
        sanitized: list[Any] = []
        if not isinstance(items, list):
            return sanitized
        for item in items:
            if not isinstance(item, Mapping):
                sanitized.append(item)
                continue
            item_type = item.get("type")
            if item_type in {"inputImage", "image", "input_image"}:
                entry: dict[str, Any] = {"type": item_type, "image_omitted": True}
                for key in ("width", "height", "mimeType", "mediaType", "detail"):
                    if key in item:
                        entry[key] = item[key]
                sanitized.append(entry)
            else:
                sanitized.append(dict(item))
        return sanitized

    def _trace(self, event: str, **fields: Any) -> None:
        if not self._trace_enabled:
            return
        self._trace_event_index += 1
        payload = {"event_index": self._trace_event_index, "event": event,
                   "time_unix": time.time(), **fields}
        try:
            self._trace_path.parent.mkdir(parents=True, exist_ok=True)
            with self._trace_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
        except OSError:
            return
        if event in {"thread_start", "turn_input", "tool_result_input", "provider_usage"}:
            summary = [f"[astra-trace] {event}"]
            if "turn_index" in fields:
                summary.append(f"turn={fields['turn_index']}")
            if "estimated_tokens" in fields:
                summary.append(f"estimated_tokens={fields['estimated_tokens']}")
            if "usage" in fields:
                summary.append(f"usage={fields['usage']}")
            summary.append(f"file={self._trace_path}")
            sys.stderr.write(" ".join(summary) + "\n")
            sys.stderr.flush()

    def start(self) -> None:
        if self._proc is not None:
            raise RuntimeError("Codex_client_already_started")
        _load_project_env()
        self._trace(
            "startup_config",
            module_file=str(Path(__file__).resolve()),
            cwd=str(Path.cwd()),
            api_key_present=bool(os.environ.get(self.api_key_environment_variable, "").strip()),
            project_env=str(Path(__file__).resolve().parents[2] / ".env"),
            project_env_exists=(Path(__file__).resolve().parents[2] / ".env").is_file(),
        )
        if not os.environ.get(self.api_key_environment_variable, "").strip():
            raise CodexClientError("astra_provider_api_key_missing")
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
            dynamic_tools_json = json.dumps(
                self.dynamic_tools, ensure_ascii=False, separators=(",", ":"), default=str
            )
            self._trace(
                "thread_start",
                model=self.model,
                reasoning_effort=self.effort,
                developer_prompt=self.developer_instructions,
                developer_chars=len(self.developer_instructions),
                developer_estimated_tokens=self._trace_estimated_tokens(self.developer_instructions),
                dynamic_tools=self.dynamic_tools,
                dynamic_tools_chars=len(dynamic_tools_json),
                dynamic_tools_estimated_tokens=self._trace_estimated_tokens(self.dynamic_tools),
                trace_path=str(self._trace_path),
            )
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
            # The upstream multi_agent_v2 path can disconnect Responses
            # streams for custom providers. Astra runs one isolated episode
            # and never needs Codex subagents, so disable it explicitly.
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
            *, idle_timeout_seconds: float = 240.0,
            max_turns_without_terminal_tool: int = 1,
            initial_input_items: Optional[list[Mapping[str, Any]]] = None) -> Dict[str, Any]:
        if self._proc is None or self.thread_id is None:
            raise CodexClientError("codex_client_not_started")
        absolute_deadline = time.monotonic() + self.max_wall_seconds
        turns = 0
        tool_calls = 0
        # A Responses turn may contain more than one function call.  That is
        # harmless for read-only calls, but it is unsafe for LIBERO actions:
        # the first execute advances the simulator and invalidates every
        # action that was planned in the same model turn.  Keep the turn alive
        # so the model can finish, but never dispatch a second action from it.
        action_tools = {
            "libero_execute_eef",
            "libero_execute_eef_chunk",
            "libero_edit_pi05_chunk",
            "libero_resume_pi05",
            "libero_stop",
        }
        refresh_tools = {"libero_observe", "pi05_propose"}
        # Only a duplicate action after an accepted action is a hard protocol
        # violation. Stale IDs and sequencing mistakes are recoverable: the
        # executor returns no_execution feedback so Astra can correct the call.
        host_recovery_errors = {
        }
        action_seen_this_turn = False
        refresh_turn_after_action = False
        host_proposal_pending = False
        restart_after_rejection = False
        recovery_rejection_seen_this_turn = False
        active_turn_id = None
        interrupt_sent_this_turn = False
        next_turn_input: Optional[dict[str, Any]] = None

        def interrupt_active_turn() -> None:
            nonlocal interrupt_sent_this_turn
            if interrupt_sent_this_turn or not self.thread_id or not active_turn_id:
                return
            try:
                self._request(
                    "turn/interrupt",
                    {"threadId": self.thread_id, "turnId": str(active_turn_id)},
                    timeout=10.0,
                )
            except CodexClientError:
                # The normal host rejection remains authoritative even if an
                # older app-server does not expose turn/interrupt.
                pass
            interrupt_sent_this_turn = True
        self._start_turn(initial_text, input_items=initial_input_items)
        turns += 1
        while time.monotonic() < absolute_deadline:
            timeout = min(float(idle_timeout_seconds), absolute_deadline - time.monotonic())
            try:
                event = self._next(timeout)
            except queue.Empty as error:
                raise CodexClientError("model_idle_timeout",
                                       model_response_received=self._model_response_received) from error
            method = event.get("method") if isinstance(event, Mapping) else None
            if method == "turn/started":
                params = event.get("params") or {}
                turn = params.get("turn") if isinstance(params, Mapping) else None
                if isinstance(turn, Mapping):
                    active_turn_id = turn.get("id") or turn.get("turnId")
                if active_turn_id is None and isinstance(params, Mapping):
                    active_turn_id = params.get("turnId")
            elif method == "item/tool/call":
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
                    if recovery_rejection_seen_this_turn:
                        # Once the host rejects a protocol call, do not let
                        # the model spend the remaining Responses turn on
                        # more invalid calls.  The host will restart a fresh
                        # turn after the current response is closed.
                        result_data = {
                            "error": "host_recovery_pending",
                            "retryable": True,
                            "no_execution": True,
                            "instruction": (
                                "The host rejected a protocol call in this Responses turn. "
                                "Do not make any more tool calls in this turn; the host will "
                                "start a fresh turn with the current state."
                            ),
                        }
                    elif name in action_tools and action_seen_this_turn:
                        # Do not let a second action from the same Responses
                        # turn reach the environment.  The first action has
                        # already advanced the simulator; a new decision must
                        # be made in a fresh turn from the host-refreshed
                        # observation/proposal.
                        result_data = {
                            "error": "stale_action_same_responses_turn",
                            "retryable": True,
                            "no_execution": True,
                            "instruction": (
                                "The previous action already executed and advanced the simulator. "
                                "This duplicate call was not executed again. Call libero_observe now, then pi05_propose, "
                                "and use the newest IDs before choosing chunk edit, execute, or resume."
                            ),
                        }
                    elif name in refresh_tools and host_proposal_pending:
                        # The host already performed observe -> pi05_propose
                        # after the previous execute.  Do not let the model
                        # create a second, stale refresh chain.
                        result_data = {
                            "error": "fresh_pi05_proposal_already_available",
                            "retryable": True,
                            "no_execution": True,
                            "instruction": (
                                "The previous action already executed. This duplicate refresh call "
                                "was not applied. Call libero_observe now, then pi05_propose, and "
                                "use the newest IDs before choosing chunk edit, execute, or resume."
                            ),
                        }
                    else:
                        result_data = dispatch(name, args)
                        if name == "libero_observe" and not bool(result_data.get("no_execution", False)):
                            # A fresh observation starts the next decision
                            # segment of the same recovery cycle.  The next
                            # execute/resume is therefore legal in this turn.
                            action_seen_this_turn = False
                        if name in action_tools and not bool(result_data.get("no_execution", False)):
                            # A recoverable validation error is feedback, not
                            # an executed action.  Keep the Responses turn
                            # alive so Astra can correct the arguments and
                            # retry.  Only a real environment action locks
                            # the action slot for this turn.
                            action_seen_this_turn = True
                            if name in {"libero_resume_pi05", "libero_stop"}:
                                host_proposal_pending = False
                    if str(result_data.get("error", "")) in host_recovery_errors:
                        restart_after_rejection = True
                        recovery_rejection_seen_this_turn = True
                        interrupt_active_turn()
                    terminal = bool(result_data.get("_terminal", False))
                    candidate_next_turn = result_data.get("_next_turn_input")
                    if isinstance(candidate_next_turn, Mapping):
                        next_turn_input = dict(candidate_next_turn)
                        refresh_turn_after_action = True
                    packet = dict(result_data)
                    packet.pop("_terminal", None)
                    packet.pop("_next_turn_input", None)
                    result = {"success": True, "contentItems": _content_items(packet)}
                except Exception as error:
                    code = getattr(error, "code", None) or getattr(error, "args", ["tool_rejected"])[0]
                    safe_code = _safe_code(code)
                    if safe_code in host_recovery_errors:
                        restart_after_rejection = True
                        recovery_rejection_seen_this_turn = True
                        interrupt_active_turn()
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
                action_seen_this_turn = False
                if restart_after_rejection:
                    # A rejected stale/repeated call is recovered by the
                    # host, not by asking Astra to remember the protocol.
                    restart_after_rejection = False
                    refresh_turn_after_action = False
                    recovery_rejection_seen_this_turn = False
                    interrupt_sent_this_turn = False
                    self._start_turn(
                        "The host rejected a stale or repeated tool call. The host owns the "
                        "recovery: use the newest host-refreshed observation and pi05 proposal, "
                        "then make exactly one current legal action call. Do not repeat the "
                        "rejected call and do not call observe or pi05_propose again unless the "
                        "host explicitly provides a new turn requiring it."
                    )
                elif refresh_turn_after_action:
                    turn_input = next_turn_input or {
                        "text": (
                            "The host refreshed the simulator state. Use the newest host "
                            "context and make exactly one current action call."
                        )
                    }
                    refresh_turn_after_action = False
                    next_turn_input = None
                    recovery_rejection_seen_this_turn = False
                    interrupt_sent_this_turn = False
                    self._start_turn(
                        str(turn_input.get("text", "Use the newest host context and make one action call.")),
                        input_items=turn_input.get("input_items"),
                    )
                else:
                    if turns >= 1 + max_turns_without_terminal_tool:
                        raise CodexClientError("model_completed_without_terminal_tool",
                                               model_response_received=self._model_response_received)
                    self._start_turn("Continue this same recovery. Use the latest tool observation, request and proposal IDs. Do not repeat completed actions. Make one next tool call.")
                    recovery_rejection_seen_this_turn = False
                    interrupt_sent_this_turn = False
                turns += 1
            elif method in ("turn/failed", "error"):
                if method == "turn/failed" and restart_after_rejection:
                    restart_after_rejection = False
                    action_seen_this_turn = False
                    refresh_turn_after_action = False
                    recovery_rejection_seen_this_turn = False
                    interrupt_sent_this_turn = False
                    self._start_turn(
                        "The previous Astra turn ended after a host-rejected stale or repeated "
                        "call. Start a new decision turn from the host's latest state and make "
                        "one current legal action call."
                    )
                    turns += 1
                    continue
                failure_code = _turn_failure_code(event) or _turn_failure_shape(event)
                suffix = "_" + failure_code if failure_code else ""
                raise CodexClientError("codex_turn_failed" + suffix,
                                       model_response_received=self._model_response_received)
        raise CodexClientError("intervention_wall_timeout",
                               model_response_received=self._model_response_received)

    def _start_turn(self, text: str,
                    *, input_items: Optional[list[Mapping[str, Any]]] = None) -> None:
        self._trace_turn_index += 1
        self._trace(
            "turn_input",
            turn_index=self._trace_turn_index,
            text=text,
            chars=len(text),
            estimated_tokens=self._trace_estimated_tokens(text),
        )
        # The app-server `turn/start` protocol expects a normal text input
        # item.  `_content_items()` is the Responses-side representation
        # (`inputText`/`inputImage`) and must not replace that text item.  Keep
        # the host context in the text field and append only image attachments
        # that the app-server accepts.  Passing `inputText` as the whole input
        # made the model receive an empty/invalid turn, so it completed without
        # emitting an action tool call.
        turn_input = [{"type": "text", "text": str(text)}]
        for item in input_items or []:
            if not isinstance(item, Mapping):
                continue
            if item.get("type") == "inputImage":
                # Dynamic-tool results use `inputImage/imageUrl`, while a
                # `turn/start` request uses the UserInput union:
                # `image/url`.  Passing the dynamic-tool shape here yields
                # app-server JSON-RPC -32600 on the next host-driven turn.
                image_url = item.get("imageUrl")
                if isinstance(image_url, str) and image_url.startswith("data:image/"):
                    turn_input.append({"type": "image", "url": image_url})
        self._request("turn/start", {
            "threadId": self.thread_id,
            "model": self.model,
            "effort": self.effort,
            "approvalPolicy": "never",
            "cwd": str(self.workspace),
            "input": turn_input,
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
        trace_items = self._trace_sanitize_content_items(result.get("contentItems"))
        self._trace(
            "tool_result_input",
            turn_index=self._trace_turn_index,
            request_id=ident,
            content_items=trace_items,
            chars=len(json.dumps(trace_items, ensure_ascii=False, separators=(",", ":"), default=str)),
            estimated_tokens=self._trace_estimated_tokens(trace_items),
        )
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
            usage = message.get("usage")
            if isinstance(usage, Mapping):
                self._trace("provider_usage", usage=dict(usage))
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
    if parsed.scheme == "https" and parsed.hostname == "api.zhizengzeng.com":
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
    """Extract only a short machine code from a failed Codex turn.

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
    # two levels below `error` (for example in codexErrorInfo). Traverse only
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
