#!/usr/bin/env python3
"""Expose the remote Taiji PhysFlow Table-2 viewer through this machine.

The Taiji container IP is protected by an ACL gateway, so a browser cannot
directly open ``http://<taiji-ip>:8080``.  This proxy serves the same static
viewer locally and forwards JSON API requests through ``tools/taiji_exec.py`` to
the running viewer inside the Taiji launcher pod.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import shlex
import subprocess
import sys
import urllib.parse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.embodied.serve_physflow_table2_canonical_viewer import _html  # noqa: E402


class Handler(SimpleHTTPRequestHandler):
    project_root: Path
    taiji_exec: Path
    task_flag: str
    instance_id: str
    host_index: str
    remote_base: str
    timeout_s: int
    three_root: Path
    mesh_root: Path

    def _send_bytes(self, data: bytes, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_file(self, path: Path) -> None:
        if not path.is_file():
            self.send_error(404, str(path))
            return
        self._send_bytes(path.read_bytes(), mimetypes.guess_type(path.name)[0] or "application/octet-stream")

    def _remote_json(self, path_qs: str) -> bytes:
        url = self.remote_base.rstrip("/") + path_qs
        # tools/taiji_exec.py appends ``; echo __EXIT_CODE__$?`` and drops any
        # line containing that marker.  curl does not append a newline for JSON,
        # so force one to keep the payload on its own line.
        cmd = f"curl -sS --max-time {self.timeout_s} {shlex.quote(url)}; printf '\\n'"
        env = os.environ.copy()
        env["TAIJI_EXEC_HOST_INDEX"] = self.host_index
        out = subprocess.check_output(
            [
                sys.executable,
                str(self.taiji_exec),
                self.task_flag,
                self.instance_id,
                cmd,
                str(self.timeout_s + 30),
                self.host_index,
            ],
            cwd=str(self.project_root),
            env=env,
            stderr=subprocess.STDOUT,
        )
        text = out.decode("utf-8", errors="replace")
        idx = text.find("{")
        if idx < 0:
            idx = text.find("[")
        if idx < 0:
            raise RuntimeError(text[-1200:])
        payload = text[idx:].strip()
        json.loads(payload)
        return payload.encode("utf-8")

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        try:
            if parsed.path in {"/", "/index.html"}:
                self._send_bytes(_html().encode("utf-8"), "text/html; charset=utf-8")
            elif parsed.path == "/favicon.ico":
                self.send_response(204)
                self.end_headers()
            elif parsed.path in {"/api/cases", "/api/case", "/api/motion"}:
                data = self._remote_json(parsed.path + (f"?{parsed.query}" if parsed.query else ""))
                self._send_bytes(data, "application/json; charset=utf-8")
            elif parsed.path.startswith("/assets/three/"):
                self._send_file(self.three_root / parsed.path.removeprefix("/assets/three/"))
            elif parsed.path.startswith("/assets/g1_mesh/"):
                self._send_file(self.mesh_root / parsed.path.removeprefix("/assets/g1_mesh/"))
            else:
                self.send_error(404)
        except subprocess.CalledProcessError as exc:
            self.send_error(502, exc.output.decode("utf-8", errors="replace")[-1800:])
        except Exception as exc:  # noqa: BLE001
            self.send_error(500, repr(exc))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--task-flag", default="task_zeyuling_20260701154254_2d380cde")
    parser.add_argument("--instance-id", default="8b1d80079f17c734019f1cea93130715")
    parser.add_argument("--host-index", default="0")
    parser.add_argument("--remote-base", default="http://127.0.0.1:8080")
    parser.add_argument("--timeout-s", type=int, default=180)
    parser.add_argument("--taiji-exec", type=Path, default=ROOT / "tools/taiji_exec.py")
    parser.add_argument("--three-root", type=Path, default=ROOT / "motion_annot_web/score_m2m/static/three")
    parser.add_argument(
        "--mesh-root",
        type=Path,
        default=ROOT / "hftrainer/models/motion/physflow/trackers/protomotions/vendor/protomotions/data/assets/mesh/G1",
    )
    args = parser.parse_args()

    Handler.project_root = ROOT
    Handler.taiji_exec = args.taiji_exec.resolve()
    Handler.task_flag = args.task_flag
    Handler.instance_id = args.instance_id
    Handler.host_index = str(args.host_index)
    Handler.remote_base = args.remote_base
    Handler.timeout_s = int(args.timeout_s)
    Handler.three_root = args.three_root.resolve()
    Handler.mesh_root = args.mesh_root.resolve()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(
        f"[physflow-table2-taiji-proxy] http://{args.host}:{args.port}/ "
        f"-> {args.task_flag}/{args.instance_id}/host{args.host_index}:{args.remote_base}",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
