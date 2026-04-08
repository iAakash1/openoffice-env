import os
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from env.core import OpenOfficeEnv

_env = None


class Handler(BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        return

    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length > 0 else b"{}"
        try:
            return json.loads(raw)
        except Exception:
            return {}

    def _send_json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send_json({"status": "OpenOfficeEnv is running"})

    def do_POST(self):
        global _env

        if self.path == "/" or self.path == "/reset":
            body = self._read_body()
            task = body.get("task", "email")
            _env = OpenOfficeEnv(task)
            obs  = _env.reset()
            self._send_json(obs.model_dump())

        elif self.path == "/step":
            if _env is None:
                self._send_json(
                    {"error": "env not initialized — call /reset first"},
                    status=400
                )
                return

            body       = self._read_body()
            action_str = body.get("action", "finish()")
            obs, reward, done, info = _env.step(action_str)

            self._send_json({
                "observation": obs.model_dump(),
                "reward":      reward.value,
                "done":        done,
                "info":        info,
            })

        else:
            self._send_json(
                {"error": f"unknown endpoint: {self.path}"},
                status=404
            )


def main():
    port = int(os.environ.get("PORT", 7860))

    try:
        server = HTTPServer(("0.0.0.0", port), Handler)
        print(f"[INFO] Server running on port {port}")
        server.serve_forever()

    except OSError as e:
        print(f"[ERROR] Failed to bind server on port {port}: {e}")

    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")