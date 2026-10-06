"""Short-lived loopback relay for the configured HTTPS Responses provider.

The relay is intentionally narrow: it binds only to loopback, accepts only the
Responses endpoint, forwards to the configured provider over HTTPS, and
records schema/status metadata rather than request or response content.
"""
from __future__ import annotations

from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import re
import threading
from typing import Any, Optional
from urllib.parse import urlsplit

import requests


_MAX_REQUEST_BYTES = 32 * 1024 * 1024
_HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailer", "transfer-encoding", "upgrade", "content-length",
    "content-encoding", "host",
}


class LocalResponsesProxy:
    """Loopback-only byte relay to one configured provider's /v1 API."""

    def __init__(self, upstream_base_url: str, *, allowed_tool_names: Optional[list[str]] = None,
                 read_timeout: float = 360.0) -> None:
        parsed = urlsplit(str(upstream_base_url).rstrip("/"))
        if (parsed.scheme != "https" or not parsed.hostname or
                parsed.username or parsed.password or parsed.query or parsed.fragment or
                parsed.port not in (None, 443) or parsed.path.rstrip("/") != "/v1"):
            raise ValueError("astra_provider_base_url_invalid")
        self.upstream_base_url = "https://" + parsed.netloc + "/v1"
        self.upstream_url = self.upstream_base_url + "/responses"
        self.allowed_tool_names = frozenset(
            name for name in (allowed_tool_names or [])
            if isinstance(name, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,128}", name)
        )
        self.read_timeout = float(read_timeout)
        self._lock = threading.Lock()
        self._request_count = 0
        self._model_catalog_reads = 0
        self._upstream_statuses: Counter[str] = Counter()
        self._event_types: Counter[str] = Counter()
        self._provider_failures: Counter[str] = Counter()
        self._request_image_items = 0
        self._request_tool_counts: Counter[str] = Counter()
        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_args: Any) -> None:
                # Default access logs can accidentally expose URLs and must not
                # be written to stdout/stderr in a robot episode.
                return

            def do_GET(self) -> None:  # noqa: N802 - stdlib handler contract
                if self.path != "/v1/models":
                    self.send_error(404)
                    return
                authorization = self.headers.get("Authorization", "")
                if not authorization.startswith("Bearer ") or not authorization[7:].strip():
                    self.send_error(401)
                    return
                headers = {
                    key: value for key, value in self.headers.items()
                    if key.lower() not in _HOP_BY_HOP and key.lower() != "accept-encoding"
                }
                headers["Accept-Encoding"] = "identity"
                try:
                    with requests.Session() as session:
                        session.trust_env = False
                        with session.get(
                            owner.upstream_base_url + "/models", headers=headers,
                            timeout=(15.0, 45.0), allow_redirects=False,
                        ) as upstream:
                            payload = upstream.content
                            with owner._lock:
                                owner._model_catalog_reads += 1
                                owner._upstream_statuses[str(upstream.status_code)] += 1
                            self.send_response(upstream.status_code)
                            content_type = upstream.headers.get("Content-Type")
                            if content_type:
                                self.send_header("Content-Type", content_type)
                            self.send_header("Content-Length", str(len(payload)))
                            self.send_header("Connection", "close")
                            self.end_headers()
                            self.wfile.write(payload)
                except (requests.RequestException, OSError):
                    try:
                        self.send_error(502, "upstream_transport_error")
                    except OSError:
                        pass

            def do_POST(self) -> None:  # noqa: N802 - stdlib handler contract
                if self.path != "/v1/responses":
                    self.send_error(404)
                    return
                raw_length = self.headers.get("Content-Length")
                try:
                    length = int(raw_length or "")
                except ValueError:
                    self.send_error(411)
                    return
                if length < 0 or length > _MAX_REQUEST_BYTES:
                    self.send_error(413)
                    return
                authorization = self.headers.get("Authorization", "")
                if not authorization.startswith("Bearer ") or not authorization[7:].strip():
                    self.send_error(401)
                    return
                body = self.rfile.read(length)
                if len(body) != length:
                    self.send_error(400)
                    return
                try:
                    body = owner._restrict_tool_schemas(body)
                except ValueError:
                    self.send_error(400, "invalid_responses_request")
                    return
                owner._record_request_body(body)
                headers = {
                    key: value for key, value in self.headers.items()
                    if key.lower() not in _HOP_BY_HOP and key.lower() != "accept-encoding"
                }
                headers["Accept-Encoding"] = "identity"
                self.close_connection = True
                with owner._lock:
                    owner._request_count += 1
                try:
                    with requests.Session() as session:
                        session.trust_env = False
                        with session.post(
                            owner.upstream_url, data=body, headers=headers, stream=True,
                            timeout=(15.0, owner.read_timeout), allow_redirects=False,
                        ) as upstream:
                            status = str(upstream.status_code)
                            with owner._lock:
                                owner._upstream_statuses[status] += 1
                            self.send_response(upstream.status_code)
                            content_type = upstream.headers.get("Content-Type")
                            if content_type:
                                self.send_header("Content-Type", content_type)
                            cache_control = upstream.headers.get("Cache-Control")
                            if cache_control:
                                self.send_header("Cache-Control", cache_control)
                            self.send_header("Connection", "close")
                            self.end_headers()
                            if upstream.status_code >= 400:
                                payload = upstream.content
                                owner._record_error_payload(payload)
                                self.wfile.write(payload)
                                self.wfile.flush()
                                return
                            pending = b""
                            for chunk in upstream.iter_content(chunk_size=8192):
                                if not chunk:
                                    continue
                                self.wfile.write(chunk)
                                self.wfile.flush()
                                pending += chunk
                                while b"\n" in pending:
                                    line, pending = pending.split(b"\n", 1)
                                    owner._record_sse_line(line)
                                if len(pending) > 1024 * 1024:
                                    pending = pending[-1024 * 1024:]
                            if pending:
                                owner._record_sse_line(pending)
                except (requests.RequestException, OSError):
                    # Never echo provider/client exceptions: they may carry
                    # endpoint, authentication, or request details.
                    try:
                        self.send_error(502, "upstream_transport_error")
                    except OSError:
                        pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        self.base_url = f"http://127.0.0.1:{self._server.server_port}/v1"

    def _record_sse_line(self, raw_line: bytes) -> None:
        line = raw_line.strip()
        if not line.startswith(b"data:"):
            return
        payload = line[5:].strip()
        if not payload or payload == b"[DONE]":
            return
        try:
            event = json.loads(payload)
        except (ValueError, UnicodeDecodeError):
            return
        if not isinstance(event, dict):
            return
        kind = _safe_label(event.get("type"))
        if kind:
            with self._lock:
                self._event_types[kind] += 1
        if kind == "response.failed":
            self._record_error_payload(payload)

    def _record_request_body(self, body: bytes) -> None:
        try:
            data = json.loads(body)
        except (ValueError, UnicodeDecodeError):
            return
        if not isinstance(data, dict):
            return
        image_count = 0
        stack = [data]
        while stack:
            item = stack.pop()
            if isinstance(item, dict):
                if item.get("type") in ("input_image", "inputImage"):
                    image_count += 1
                stack.extend(item.values())
            elif isinstance(item, list):
                stack.extend(item)
        names = []
        tools = data.get("tools")
        if isinstance(tools, list):
            for item in tools:
                if isinstance(item, dict):
                    name = _safe_label(item.get("name"))
                    if name:
                        names.append(name)
        with self._lock:
            self._request_image_items += image_count
            for name in names:
                self._request_tool_counts[name] += 1

    def _restrict_tool_schemas(self, body: bytes) -> bytes:
        try:
            data = json.loads(body)
        except (ValueError, UnicodeDecodeError) as error:
            raise ValueError("invalid_responses_request") from error
        if not isinstance(data, dict):
            raise ValueError("invalid_responses_request")
        tools = data.get("tools")
        if tools is None:
            return body
        if not isinstance(tools, list):
            raise ValueError("invalid_responses_request")
        data["tools"] = [
            item for item in tools
            if isinstance(item, dict) and item.get("type") == "function" and
            item.get("name") in self.allowed_tool_names
        ]
        return json.dumps(data, ensure_ascii=False, separators=(",", ":"),
                          allow_nan=False).encode("utf-8")

    def _record_error_payload(self, payload: bytes) -> None:
        try:
            data = json.loads(payload)
        except (ValueError, UnicodeDecodeError):
            return
        if not isinstance(data, dict):
            return
        error = data.get("error")
        if isinstance(error, dict):
            label = _safe_label(error.get("type")) or "unknown"
            code = _safe_label(error.get("code")) or "unknown"
            with self._lock:
                self._provider_failures[f"{label}:{code}"] += 1

    def summary(self) -> dict[str, Any]:
        with self._lock:
            return {
                "request_count": self._request_count,
                "model_catalog_reads": self._model_catalog_reads,
                "upstream_statuses": dict(self._upstream_statuses),
                "event_types": dict(self._event_types),
                "provider_failures": dict(self._provider_failures),
                "input_image_items": self._request_image_items,
                "tool_names_seen": sorted(self._request_tool_counts),
            }

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2.0)

    def __enter__(self) -> "LocalResponsesProxy":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()


def _safe_label(value: Any) -> Optional[str]:
    if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.:-]{1,64}", value):
        return value
    return None
