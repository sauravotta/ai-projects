#!/usr/bin/env python3
"""Local standard-library web UI for the durable support demo."""

import argparse
import hmac
import html
import ipaddress
import logging
import os
import re
import secrets
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, quote, urlsplit


LOG = logging.getLogger("durable-state.web")
MAX_BODY_BYTES = 8192
MAX_REASON_LENGTH = 500
MAX_IDENTIFIER_LENGTH = 128
IDENTIFIER_PATTERN = re.compile(r"[A-Za-z0-9._-]{1,128}\Z")
RUNNABLE_ITEM_STATUSES = {"queued", "processing", "approved", "ready_action"}
DEFAULT_MODEL = "nvidia/nemotron-3.5-lightning:free"


def _text(value):
    return html.escape("" if value is None else str(value), quote=True)


def _valid_identifier(value):
    return (
        isinstance(value, str)
        and len(value) <= MAX_IDENTIFIER_LENGTH
        and value not in (".", "..")
        and IDENTIFIER_PATTERN.fullmatch(value) is not None
    )


def make_handler(store, manager, csrf_token, startup_warnings=None):
    startup_warnings = startup_warnings or {}

    class DurableStateHandler(BaseHTTPRequestHandler):
        server_version = "DurableStateDemo"
        sys_version = ""

        def _send(self, status, body=b"", content_type="text/html; charset=utf-8",
                  headers=None):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'none'; style-src 'unsafe-inline'; "
                "form-action 'self'; base-uri 'none'; frame-ancestors 'none'",
            )
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Connection", "close")
            if headers:
                for name, value in headers:
                    self.send_header(name, value)
            self.end_headers()
            if body:
                self.wfile.write(body)
            self.close_connection = True

        def _page(self, status, title, content):
            page = (
                "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
                "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
                f"<title>{_text(title)}</title>"
                "<style>body{font:16px system-ui,sans-serif;max-width:60rem;"
                "margin:2rem auto;padding:0 1rem;color:#222}table{border-collapse:"
                "collapse;width:100%}th,td{text-align:left;padding:.5rem;border:"
                "1px solid #ccc;vertical-align:top}form{margin:.75rem 0}textarea{"
                "display:block;width:min(100%,36rem);min-height:4rem}pre{white-space:"
                "pre-wrap;overflow-wrap:anywhere}button{padding:.4rem .8rem}</style>"
                "</head><body>"
                f"<h1>{_text(title)}</h1>{content}</body></html>"
            ).encode("utf-8")
            self._send(status, page)

        def _error(self, status, message):
            self._page(status, "Request error", f"<p>{_text(message)}</p><p><a href=\"/\">"
                                                  "Back to batches</a></p>")

        def _csrf_input(self):
            return f'<input type="hidden" name="csrf_token" value="{_text(csrf_token)}">'

        def _read_form(self):
            lengths = self.headers.get_all("Content-Length", [])
            if len(lengths) != 1 or not lengths[0].isdigit():
                self.close_connection = True
                self._error(411, "A valid Content-Length header is required.")
                return None
            length = int(lengths[0])
            if length > MAX_BODY_BYTES:
                self.close_connection = True
                self._error(413, "Request body is too large.")
                return None
            if self.headers.get("Transfer-Encoding"):
                self.close_connection = True
                self._error(400, "Transfer-Encoding is not supported.")
                return None
            content_type = self.headers.get("Content-Type", "")
            if content_type.split(";", 1)[0].strip().lower() != \
                    "application/x-www-form-urlencoded":
                self.close_connection = True
                self._error(415, "Expected a URL-encoded form.")
                return None
            try:
                raw_body = self.rfile.read(length)
                if len(raw_body) != length:
                    raise ValueError("incomplete body")
                body = raw_body.decode("utf-8", errors="strict")
                form = parse_qs(
                    body,
                    keep_blank_values=True,
                    strict_parsing=True,
                    max_num_fields=5,
                    encoding="utf-8",
                    errors="strict",
                )
            except (UnicodeDecodeError, ValueError):
                self.close_connection = True
                self._error(400, "Malformed form data.")
                return None
            tokens = form.get("csrf_token", [])
            if len(tokens) != 1 or not hmac.compare_digest(tokens[0], csrf_token):
                self._error(403, "Invalid or missing CSRF token.")
                return None
            return form

        @staticmethod
        def _single_value(form, name):
            values = form.get(name, [])
            if len(values) != 1:
                return None
            return values[0]

        def do_GET(self):
            parsed = urlsplit(self.path)
            if parsed.path == "/":
                self._list_batches()
            elif parsed.path.startswith("/batch/"):
                batch_id = parsed.path[len("/batch/"):]
                if "/" not in batch_id and _valid_identifier(batch_id):
                    self._show_batch(batch_id)
                else:
                    self._error(404, "Batch not found.")
            else:
                self._error(404, "Page not found.")

        def _list_batches(self):
            try:
                batches = store.list_batches()
            except Exception:
                LOG.exception("Could not list batches")
                self._error(500, "Could not load batches.")
                return

            rows = []
            for batch in batches:
                batch_id = batch.get("id")
                status = batch.get("status")
                if not _valid_identifier(batch_id):
                    continue
                encoded_id = quote(batch_id, safe="")
                rows.append(
                    "<tr><td><a href=\"/batch/"
                    f"{_text(encoded_id)}\">{_text(batch_id)}</a></td>"
                    f"<td>{_text(status)}</td></tr>"
                )
            batch_rows = "".join(rows) or (
                '<tr><td colspan="2">No batches yet.</td></tr>'
            )
            table = (
                "<table><thead><tr><th>Batch</th><th>Status</th></tr></thead>"
                f"<tbody>{batch_rows}"
                "</tbody></table>"
            )
            form = (
                "<form method=\"post\" action=\"/batches\">"
                f"{self._csrf_input()}<button type=\"submit\">Create batch</button></form>"
            )
            warnings = "".join(
                f"<p role=\"alert\">Batch {_text(batch_id)} was not resumed: "
                f"{_text(message)}</p>"
                for batch_id, message in startup_warnings.items()
            )
            self._page(200, "Durable support batches", warnings + form + table)

        def _show_batch(self, batch_id):
            try:
                batch = store.get_batch(batch_id)
            except (KeyError, ValueError):
                self._error(404, "Batch not found.")
                return
            except Exception:
                LOG.exception("Could not load batch")
                self._error(500, "Could not load batch.")
                return

            items_html = []
            for item in batch.get("items", []):
                fields = (
                    ("ID", item.get("id")),
                    ("Status", item.get("status")),
                    ("Risk", item.get("risk")),
                    ("Confidence", item.get("confidence")),
                    ("Worker attempts", item.get("attempts")),
                    ("Text", item.get("text")),
                    ("Rationale", item.get("rationale")),
                    ("Action", item.get("action")),
                    ("Error", item.get("error")),
                    ("Decision", item.get("decision")),
                    ("Reason", item.get("reason")),
                )
                details = "".join(
                    f"<dt>{_text(label)}</dt><dd>{_text(value)}</dd>"
                    for label, value in fields
                    if value is not None and value != ""
                )
                item_id = item.get("id")
                review_form = ""
                if item.get("status") == "waiting_review" and \
                        _valid_identifier(item_id):
                    encoded_item_id = quote(item_id, safe="")
                    review_form = (
                        f"<form method=\"post\" action=\"/items/{_text(encoded_item_id)}"
                        "/review\">"
                        f"{self._csrf_input()}"
                        "<label>Optional reason: <textarea name=\"reason\" "
                        f"maxlength=\"{MAX_REASON_LENGTH}\"></textarea></label>"
                        "<button type=\"submit\" name=\"decision\" value=\"approve\">"
                        "Approve</button> "
                        "<button type=\"submit\" name=\"decision\" value=\"reject\">"
                        "Reject</button></form>"
                    )
                items_html.append(
                    f"<section><h2>Item {_text(item_id)}</h2><dl>{details}</dl>"
                    f"{review_form}</section>"
                )

            warning = (
                f"<p role=\"alert\">{_text(startup_warnings[batch_id])}</p>"
                if batch_id in startup_warnings
                else ""
            )
            item_details = "".join(items_html) or "<p>This batch has no items.</p>"
            events_html = "".join(
                f"<li>{_text(event.get('created_at'))} | {_text(event.get('item_id'))} | "
                f"{_text(event.get('event'))}: {_text(event.get('details'))}</li>"
                for event in batch.get("events", [])
            )
            content = (
                f"<p><a href=\"/\">All batches</a></p><p>Status: "
                f"{_text(batch.get('status'))}</p>{warning}{item_details}"
                f"<h2>Audit events</h2><ol>{events_html}</ol>"
            )
            self._page(200, f"Batch {_text(batch.get('id', batch_id))}", content)

        def do_POST(self):
            parsed = urlsplit(self.path)
            if parsed.path not in ("/batches",) and not parsed.path.startswith("/items/"):
                self._error(404, "Page not found.")
                return
            form = self._read_form()
            if form is None:
                return
            if parsed.path == "/batches":
                self._create_batch()
                return

            prefix, suffix = "/items/", "/review"
            item_path = parsed.path[len(prefix):] if parsed.path.startswith(prefix) else ""
            if not item_path.endswith(suffix):
                self._error(404, "Page not found.")
                return
            item_id = item_path[:-len(suffix)]
            if "/" in item_id or not _valid_identifier(item_id):
                self._error(404, "Item not found.")
                return
            self._review_item(item_id, form)

        def _create_batch(self):
            try:
                batch_id = store.create_batch()
                if not _valid_identifier(batch_id):
                    raise RuntimeError("store returned an invalid batch identifier")
                manager.start(batch_id)
            except Exception:
                LOG.exception("Could not create or start batch")
                self._error(
                    500,
                    "Could not create or start the batch. Check the server log for details.",
                )
                return
            self._redirect("/batch/" + quote(batch_id, safe=""))

        def _review_item(self, item_id, form):
            decision = self._single_value(form, "decision")
            reasons = form.get("reason", [""])
            reason = reasons[0] if len(reasons) == 1 else None
            if decision not in ("approve", "reject"):
                self._error(400, "Decision must be approve or reject.")
                return
            if reason is None:
                self._error(400, "Provide at most one reason field.")
                return
            if len(reason) > MAX_REASON_LENGTH:
                self._error(400, "Reason is too long.")
                return

            try:
                batch_id = store.review(item_id, decision, reason)
                if not _valid_identifier(batch_id):
                    raise RuntimeError("store returned an invalid batch identifier")
            except KeyError:
                self._error(404, "Item not found.")
                return
            except ValueError:
                self._error(409, "This item cannot be reviewed in its current state.")
                return
            except Exception:
                LOG.exception("Could not review item")
                self._error(500, "Could not save the review.")
                return

            try:
                manager.start(batch_id)
            except ValueError:
                self._error(
                    409,
                    "Review was saved, but this batch's stored mode/model is "
                    "incompatible with the server. Restart with the original "
                    "mode and model; do not submit the review again.",
                )
                return
            except Exception:
                LOG.exception("Review was saved but batch could not be started")
                self._error(
                    500,
                    "Review was saved, but the batch could not be started. "
                    "Check the server log for details.",
                )
                return
            self._redirect("/batch/" + quote(batch_id, safe=""))

        def _redirect(self, location):
            self._send(
                303,
                b"",
                headers=(("Location", location),),
            )

        def do_PUT(self):
            self._error(405, "Method not allowed.")

        def do_PATCH(self):
            self._error(405, "Method not allowed.")

        def do_DELETE(self):
            self._error(405, "Method not allowed.")

        def log_message(self, fmt, *args):
            LOG.info("%s - %s", self.client_address[0], fmt % args)

    return DurableStateHandler


def _is_loopback_host(host):
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.split("%", 1)[0]).is_loopback
    except ValueError:
        return False


def _resume_batches(store, manager):
    warnings = {}
    try:
        batches = store.list_batches()
    except Exception:
        LOG.exception("Could not inspect existing batches for recovery")
        return warnings

    for batch in batches:
        batch_id = batch.get("id")
        if not _valid_identifier(batch_id):
            continue
        try:
            details = store.get_batch(batch_id)
        except Exception:
            LOG.exception("Could not inspect batch %s for recovery", batch_id)
            continue
        if not any(
            item.get("status") in RUNNABLE_ITEM_STATUSES
            for item in details.get("items", [])
        ):
            continue
        try:
            manager.start(batch_id)
        except ValueError:
            message = (
                "Its stored mode/model is incompatible with this server. "
                "Restart the server with the mode and model originally used for this batch."
            )
            warnings[batch_id] = message
            LOG.warning("Batch %s was not resumed. %s", batch_id, message)
        except Exception:
            message = "It could not be resumed; check the server log for details."
            warnings[batch_id] = message
            LOG.exception("Could not resume batch %s", batch_id)
    return warnings


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="durable_state.sqlite3", help="SQLite database path")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: localhost)")
    parser.add_argument("--port", type=int, default=8000, help="TCP port (default: 8000)")
    parser.add_argument("--mode", choices=("offline", "openrouter"), default="offline")
    parser.add_argument("--max-concurrency", type=int, default=3)
    parser.add_argument(
        "--model",
        default=None,
        help="Model name for openrouter mode",
    )
    args = parser.parse_args(argv)
    if not 0 <= args.port <= 65535:
        parser.error("--port must be between 0 and 65535")
    if args.max_concurrency < 1:
        parser.error("--max-concurrency must be at least 1")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if not _is_loopback_host(args.host):
        LOG.warning(
            "Binding to a non-loopback interface exposes this demo to the network."
        )

    try:
        from workflow import Manager, Store

        if args.mode == "openrouter":
            from dotenv import load_dotenv
            from pathlib import Path
            load_dotenv(Path(__file__).resolve().parent / ".env")
        model = args.model or os.getenv("OPENROUTER_MODEL") or DEFAULT_MODEL
        store = Store(args.db)
        manager = Manager(store, args.mode, args.max_concurrency, model)
        startup_warnings = _resume_batches(store, manager)
        handler = make_handler(
            store, manager, secrets.token_urlsafe(32), startup_warnings
        )
        server = HTTPServer((args.host, args.port), handler)
    except Exception:
        LOG.exception("Could not initialize the durable support web UI")
        return 1

    host, port = server.server_address[:2]
    print(f"Serving durable support UI at http://{host}:{port}/", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOG.info("Shutting down")
    except Exception:
        LOG.exception("HTTP server stopped unexpectedly")
        return 1
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
