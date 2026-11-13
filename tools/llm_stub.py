#!/usr/bin/env python3
"""Offline LLM stub for demos and tests.

Serves pre-authored responses over an OpenAI-compatible
``/v1/chat/completions`` endpoint, so FolderLake's real LLM tier
(provider layer, prompt packaging, JSON parsing, hash caching) runs
end-to-end without a live model. The GitHub Pages demo was produced
exactly this way, with Muse as the "model": the responses file maps
each demo file's name to the tags / summary / entities Muse wrote
for it.

Usage:
    python tools/llm_stub.py responses.json [port]     # default 8899

    FOLDERLAKE_LLM_PROVIDER=openai \
    FOLDERLAKE_LLM_BASE_URL=http://127.0.0.1:8899/v1 \
    FOLDERLAKE_LLM_MODEL=muse \
        folderlake scan <folder> --llm

Every request is logged to stderr; an unknown filename gets an empty
reply, which the pipeline treats as an LLM failure (rule fallback),
so coverage gaps are visible in the scan stats.
"""

from __future__ import annotations

import json
import re
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

RESPONSES: dict[str, dict] = {}
_NAME_RE = re.compile(r"Filename: (.+)")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # keep stderr for our own log
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
            prompt = body["messages"][0]["content"]
            name = _NAME_RE.search(prompt).group(1).strip()
        except Exception:
            name = ""
        entry = RESPONSES.get(name)
        print(f"stub: {name!r} -> {'hit' if entry else 'MISS'}",
              file=sys.stderr, flush=True)
        content = json.dumps(entry, ensure_ascii=False) if entry else ""
        payload = {
            "id": "chatcmpl-stub",
            "object": "chat.completion",
            "model": "muse",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0,
                      "total_tokens": 0},
        }
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    global RESPONSES
    RESPONSES = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8899
    print(f"stub: {len(RESPONSES)} responses on http://127.0.0.1:{port}/v1",
          file=sys.stderr, flush=True)
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
