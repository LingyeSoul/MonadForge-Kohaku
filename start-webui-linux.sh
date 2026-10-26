#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# Detect Python interpreter: prefer .venv/bin/python (Linux venv), then system python
if [ -x ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
elif command -v python3 &>/dev/null; then
    PYTHON="python3"
else
    PYTHON="python"
fi

# The WebUI serves the built Vue app from webui/frontend/dist. Without that
# directory FastAPI exposes only /api/* and every page load is a bare 404, so
# build it on the first run (setup-linux.sh step 4 does the same) instead of
# opening a broken tab.
DIST_INDEX="webui/frontend/dist/index.html"
if [ ! -f "$DIST_INDEX" ]; then
    if ! command -v node &>/dev/null && [ -x "tools/node/bin/node" ]; then
        export PATH="$PWD/tools/node/bin:$PATH"
    fi
    if ! command -v node &>/dev/null; then
        echo "[ERROR] WebUI frontend is not built ($DIST_INDEX is missing) and Node.js is not installed." >&2
        echo "        Without it http://127.0.0.1:8000/ returns 404 (only /api/* is served)." >&2
        echo "        Run ./setup-linux.sh (installs portable Node.js and builds), or install Node.js then:" >&2
        echo "          cd webui/frontend && npm install && npm run build" >&2
        exit 1
    fi
    echo "WebUI frontend not built yet — building (npm ci + vite build) ..."
    if ! ( cd webui/frontend \
        && node ../../scripts/webui/ensure_npm_deps.mjs --ensure --clean \
        && npm run build ); then
        echo "[ERROR] Frontend build failed; see the output above." >&2
        exit 1
    fi
fi

WEBUI_URL="http://127.0.0.1:8000"
ANALYZER_URL="http://127.0.0.1:8320"

echo "Starting MonadForge-Kohaku WebUI..."
echo "  WebUI    -> $WEBUI_URL"
echo "  Analyzer -> $ANALYZER_URL"

"$PYTHON" -m webui "$@" &
WEBUI_PID=$!

"$PYTHON" -m scripts.run_analyzer.server --port 8320 &
ANALYZER_PID=$!

cleanup() {
    kill "$ANALYZER_PID" 2>/dev/null || true
    kill "$WEBUI_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

sleep 3
if command -v xdg-open &>/dev/null; then
    setsid xdg-open "$WEBUI_URL" 2>/dev/null || true
    setsid xdg-open "$ANALYZER_URL" 2>/dev/null || true
elif command -v open &>/dev/null; then
    open "$WEBUI_URL" 2>/dev/null || true
    open "$ANALYZER_URL" 2>/dev/null || true
fi

echo "WebUI PID: $WEBUI_PID / Analyzer PID: $ANALYZER_PID (Ctrl+C to stop)"
# The two processes form one supervised unit. If either sidecar exits, return
# its status immediately; the EXIT trap terminates the survivor.
wait -n "$WEBUI_PID" "$ANALYZER_PID"
