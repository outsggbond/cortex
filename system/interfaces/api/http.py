# -*- coding: utf-8 -*-
"""Thin JSON/HTTP shell wired to the interface adapters."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict
from urllib.parse import parse_qs, urlparse

from .automation import AutomationAPIAdapter, create_automation_api
from .chat import ChatAPIAdapter, create_chat_api
from .computer import ComputerUseAPIAdapter, create_computer_use_api


logger = logging.getLogger(__name__)


def _default_headers() -> Dict[str, str]:
    return {
        "Content-Type": "application/json; charset=utf-8",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "Content-Type",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    }


@dataclass
class HTTPAPIRequest:
    method: str
    path: str
    body: Dict[str, Any] | None = None
    query: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HTTPAPIResponse:
    status_code: int
    body: Any = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=_default_headers)

    def to_bytes(self) -> bytes:
        if int(self.status_code) == 204:
            return b""
        return json.dumps(self.body if self.body is not None else {}, ensure_ascii=False).encode("utf-8")


class HTTPAPIShell:
    def __init__(
        self,
        args: Any,
        *,
        chat_adapter: ChatAPIAdapter | None = None,
        computer_adapter: ComputerUseAPIAdapter | None = None,
        automation_adapter: AutomationAPIAdapter | None = None,
    ) -> None:
        self.args = args
        self._chat_adapter = chat_adapter
        self._computer_adapter = computer_adapter
        self._automation_adapter = automation_adapter

    @property
    def chat_adapter(self) -> ChatAPIAdapter:
        if self._chat_adapter is None:
            self._chat_adapter = create_chat_api(self.args)
        return self._chat_adapter

    @property
    def computer_adapter(self) -> ComputerUseAPIAdapter:
        if self._computer_adapter is None:
            self._computer_adapter = create_computer_use_api(self.args)
        return self._computer_adapter

    @property
    def automation_adapter(self) -> AutomationAPIAdapter:
        if self._automation_adapter is None:
            self._automation_adapter = create_automation_api(self.args)
        return self._automation_adapter

    def list_routes(self) -> list[dict[str, str]]:
        return [
            {"method": "GET", "path": "/healthz", "name": "health"},
            {"method": "POST", "path": "/api/chat/respond", "name": "chat.respond"},
            {"method": "POST", "path": "/api/computer/run", "name": "computer.run"},
            {"method": "GET", "path": "/api/automation/workflows", "name": "automation.list"},
            {"method": "POST", "path": "/api/automation/scan", "name": "automation.scan"},
            {"method": "POST", "path": "/api/automation/run", "name": "automation.run"},
            {"method": "POST", "path": "/api/automation/loop", "name": "automation.loop"},
        ]

    def handle(self, request: HTTPAPIRequest) -> HTTPAPIResponse:
        method = str(request.method or "GET").strip().upper() or "GET"
        path = self._normalize_path(request.path)
        routes = {
            ("GET", "/healthz"): self._handle_health,
            ("GET", "/api/healthz"): self._handle_health,
            ("POST", "/api/chat/respond"): self._handle_chat_respond,
            ("POST", "/api/computer/run"): self._handle_computer_run,
            ("GET", "/api/automation/workflows"): self._handle_automation_workflows,
            ("POST", "/api/automation/scan"): self._handle_automation_scan,
            ("POST", "/api/automation/run"): self._handle_automation_run,
            ("POST", "/api/automation/loop"): self._handle_automation_loop,
        }
        if method == "OPTIONS":
            return HTTPAPIResponse(status_code=204, body=None)

        handler = routes.get((method, path))
        if handler is None:
            allowed = sorted({verb for (verb, route_path) in routes if route_path == path})
            if allowed:
                return HTTPAPIResponse(
                    status_code=405,
                    body={
                        "ok": False,
                        "error": f"method_not_allowed: {method} {path}",
                        "allowed_methods": allowed,
                    },
                )
            return HTTPAPIResponse(
                status_code=404,
                body={"ok": False, "error": f"not_found: {path}"},
            )

        try:
            return handler(request)
        except ValueError as exc:
            return HTTPAPIResponse(
                status_code=400,
                body={"ok": False, "error": str(exc)},
            )
        except Exception as exc:
            logger.exception("http api shell failed for %s %s", method, path)
            return HTTPAPIResponse(
                status_code=500,
                body={"ok": False, "error": str(exc) or "internal_server_error"},
            )

    def _normalize_path(self, path: str) -> str:
        raw = "/" + str(path or "").strip().lstrip("/")
        return raw.rstrip("/") or "/"

    def _handle_health(self, _request: HTTPAPIRequest) -> HTTPAPIResponse:
        return HTTPAPIResponse(
            status_code=200,
            body={
                "ok": True,
                "service": "system.interfaces.api.http",
                "routes": self.list_routes(),
            },
        )

    def _handle_chat_respond(self, request: HTTPAPIRequest) -> HTTPAPIResponse:
        payload = dict(request.body or {})
        kwargs: Dict[str, Any] = {
            "history": payload.get("history"),
        }
        if payload.get("history_limit") is not None:
            kwargs["history_limit"] = int(payload.get("history_limit"))
        result = self.chat_adapter.respond(str(payload.get("user_text", "") or ""), **kwargs)
        return HTTPAPIResponse(status_code=200, body=result.to_dict())

    def _handle_computer_run(self, request: HTTPAPIRequest) -> HTTPAPIResponse:
        payload = dict(request.body or {})
        return HTTPAPIResponse(
            status_code=200,
            body=self.computer_adapter.run_goal(payload.get("goal")),
        )

    def _handle_automation_workflows(self, _request: HTTPAPIRequest) -> HTTPAPIResponse:
        return HTTPAPIResponse(
            status_code=200,
            body={"workflows": self.automation_adapter.list_workflows()},
        )

    def _handle_automation_scan(self, request: HTTPAPIRequest) -> HTTPAPIResponse:
        payload = dict(request.body or {})
        return HTTPAPIResponse(
            status_code=200,
            body=self.automation_adapter.scan(
                workflow_ids=payload.get("workflow_ids"),
                force_run=bool(payload.get("force_run", False)),
            ),
        )

    def _handle_automation_run(self, request: HTTPAPIRequest) -> HTTPAPIResponse:
        payload = dict(request.body or {})
        return HTTPAPIResponse(
            status_code=200,
            body=self.automation_adapter.run(payload.get("workflow_ids") or []),
        )

    def _handle_automation_loop(self, request: HTTPAPIRequest) -> HTTPAPIResponse:
        payload = dict(request.body or {})
        return HTTPAPIResponse(
            status_code=200,
            body=self.automation_adapter.loop(
                poll_s=payload.get("poll_s"),
                max_cycles=payload.get("max_cycles"),
                workflow_ids=payload.get("workflow_ids"),
            ),
        )


def build_http_api_handler(shell: HTTPAPIShell) -> type[BaseHTTPRequestHandler]:
    class _HTTPAPIHandler(BaseHTTPRequestHandler):
        server_version = "InterfacesHTTP/1.0"

        def do_GET(self) -> None:  # noqa: N802
            self._dispatch()

        def do_POST(self) -> None:  # noqa: N802
            self._dispatch()

        def do_OPTIONS(self) -> None:  # noqa: N802
            self._dispatch()

        def log_message(self, format: str, *args: Any) -> None:
            logger.info("http api shell | " + format, *args)

        def _dispatch(self) -> None:
            parsed = urlparse(self.path)
            payload: Dict[str, Any] | None = None
            if self.command in {"POST", "PUT", "PATCH"}:
                content_length = int(self.headers.get("Content-Length", "0") or 0)
                raw_body = self.rfile.read(content_length) if content_length > 0 else b""
                if raw_body:
                    try:
                        decoded = json.loads(raw_body.decode("utf-8"))
                    except json.JSONDecodeError as exc:
                        self._write_response(
                            HTTPAPIResponse(
                                status_code=400,
                                body={"ok": False, "error": f"invalid_json: {exc.msg}"},
                            )
                        )
                        return
                    if decoded is not None and not isinstance(decoded, dict):
                        self._write_response(
                            HTTPAPIResponse(
                                status_code=400,
                                body={"ok": False, "error": "json body must be an object"},
                            )
                        )
                        return
                    payload = decoded

            query = {
                key: values if len(values) > 1 else values[0]
                for key, values in parse_qs(parsed.query, keep_blank_values=True).items()
            }
            response = shell.handle(
                HTTPAPIRequest(
                    method=self.command,
                    path=parsed.path,
                    body=payload,
                    query=query,
                )
            )
            self._write_response(response)

        def _write_response(self, response: HTTPAPIResponse) -> None:
            body = response.to_bytes()
            self.send_response(int(response.status_code))
            for key, value in dict(response.headers or {}).items():
                self.send_header(str(key), str(value))
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if body:
                self.wfile.write(body)

    return _HTTPAPIHandler


def create_http_api_shell(
    args: Any,
    *,
    chat_adapter: ChatAPIAdapter | None = None,
    computer_adapter: ComputerUseAPIAdapter | None = None,
    automation_adapter: AutomationAPIAdapter | None = None,
) -> HTTPAPIShell:
    return HTTPAPIShell(
        args,
        chat_adapter=chat_adapter,
        computer_adapter=computer_adapter,
        automation_adapter=automation_adapter,
    )


def run_http_api_server(
    args: Any,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> None:
    shell = create_http_api_shell(args)
    server = ThreadingHTTPServer((str(host or "127.0.0.1"), int(port)), build_http_api_handler(shell))
    logger.info("HTTP API shell listening on http://%s:%s", host, port)
    with server:
        server.serve_forever()


__all__ = [
    "HTTPAPIRequest",
    "HTTPAPIResponse",
    "HTTPAPIShell",
    "build_http_api_handler",
    "create_http_api_shell",
    "run_http_api_server",
]
