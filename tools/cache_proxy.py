#!/usr/bin/env python3
"""Cache-friendly front for a local Splash server.

    cache_proxy.py LISTEN_PORT UPSTREAM_PORT -- ENGINE_COMMAND...

Starts ENGINE_COMMAND (a server on UPSTREAM_PORT) and forwards every request to
it unchanged, except one reordering of Anthropic /v1/messages bodies: Claude Code
sends a system message (environment, agent and skill listings) right after the
first user message, so whatever the user types sits before ~2-3K stable tokens
and the server's prefix cache can never cover them on a session's first turn.
The proxy moves those system messages ahead of the first user message. The rule
looks only at the conversation's opening, so every later turn of the session
renders the same prefix. Stopping the proxy stops the engine.
"""

import http.client
import http.server
import json
import signal
import subprocess
import sys
import threading

HOP_HEADERS = {"connection", "keep-alive", "transfer-encoding", "content-length", "host"}


def reorder(body):
    """Moves the system messages that directly follow the first user message
    ahead of it; leaves anything else, and anything unparseable, alone."""
    try:
        request = json.loads(body)
    except ValueError:
        return body
    messages = request.get("messages") if isinstance(request, dict) else None
    if not isinstance(messages, list) or len(messages) < 2:
        return body
    first = messages[0]
    if not isinstance(first, dict) or first.get("role") != "user":
        return body
    if isinstance(first.get("content"), list) and any(
            isinstance(part, dict) and part.get("type") == "tool_result" for part in first["content"]):
        return body
    count = 0
    while 1 + count < len(messages) and isinstance(messages[1 + count], dict) \
            and messages[1 + count].get("role") == "system":
        count += 1
    if not count:
        return body
    request["messages"] = messages[1:1 + count] + [first] + messages[1 + count:]
    return json.dumps(request).encode()


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    upstream_port = 0

    def forward(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else None
        if body is not None and self.command == "POST" and self.path.split("?")[0].startswith("/v1/messages"):
            body = reorder(body)
        headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP_HEADERS}
        if body is not None:
            headers["Content-Length"] = str(len(body))
        upstream = http.client.HTTPConnection("127.0.0.1", self.upstream_port)
        try:
            try:
                upstream.request(self.command, self.path, body=body, headers=headers)
                response = upstream.getresponse()
            except OSError:
                self.send_error(502, "engine not reachable")
                return
            self.send_response(response.status, response.reason)
            for key, value in response.getheaders():
                if key.lower() not in HOP_HEADERS:
                    self.send_header(key, value)
            if self.command == "HEAD":
                self.send_header("Content-Length", response.getheader("Content-Length", "0"))
                self.end_headers()
                return
            # Streamed (SSE) bodies are re-chunked as they arrive.
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            while True:
                data = response.read1(65536)
                if not data:
                    break
                self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            # The client went away; closing upstream cancels the engine's work.
            self.close_connection = True
        finally:
            upstream.close()

    do_GET = do_POST = do_HEAD = do_PUT = do_DELETE = do_OPTIONS = forward

    def log_message(self, *args):
        pass


class Server(http.server.ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):
        # A client dropping a kept-alive connection is routine, not an error.
        if not isinstance(sys.exc_info()[1], (BrokenPipeError, ConnectionResetError)):
            super().handle_error(request, client_address)


def main():
    if len(sys.argv) < 5 or sys.argv[3] != "--":
        sys.exit(__doc__)
    listen, Handler.upstream_port = int(sys.argv[1]), int(sys.argv[2])
    engine = subprocess.Popen(sys.argv[4:])
    server = Server(("127.0.0.1", listen), Handler)

    def stop(*_):
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    threading.Thread(target=lambda: (engine.wait(), stop()), daemon=True).start()
    server.serve_forever()
    if engine.poll() is None:
        engine.send_signal(signal.SIGINT)
        try:
            engine.wait(timeout=60)
        except subprocess.TimeoutExpired:
            engine.kill()
    sys.exit(engine.returncode or 0)


if __name__ == "__main__":
    main()
