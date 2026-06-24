#!/usr/bin/env python3
"""Minimal static server for the MotionFix Three.js viewer.

Forces JS/JSON MIME types so ES module imports work, and serves the directory
containing this script. Run from anywhere:

    /usr/bin/python3 web_viz/motionfix/serve.py --port 8123
"""
import argparse
import os
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


class Handler(SimpleHTTPRequestHandler):
    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".js": "application/javascript",
        ".mjs": "application/javascript",
        ".json": "application/json",
        ".html": "text/html",
    }

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, *a):  # quiet
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8123)
    args = ap.parse_args()
    root = os.path.dirname(os.path.abspath(__file__))
    httpd = ThreadingHTTPServer((args.host, args.port), partial(Handler, directory=root))
    print(f"MotionFix Three.js viewer: http://{args.host}:{args.port}/  (root={root})")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
