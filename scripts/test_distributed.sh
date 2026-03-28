#!/usr/bin/env bash
#
# End-to-end distributed inference test.
#
# Usage (single machine, 2 processes):
#   ./scripts/test_distributed.sh localhost
#
# Usage (cross-machine, run on coordinator machine):
#   ./scripts/test_distributed.sh cross-machine <worker_host>
#
# Prerequisites:
#   - GGUF model at $FRACTURE_MODEL_PATH (or models/llama-3.1-8b-instruct-f16.gguf)
#   - tokenizer.json in the same directory as the model
#   - For cross-machine: worker binary deployed on the remote host
#   - cargo build --release completed

set -euo pipefail

MODE="${1:-localhost}"
WORKER_HOST="${2:-}"
MODEL_PATH="${FRACTURE_MODEL_PATH:-models/llama-3.1-8b-instruct-f16.gguf}"
COORD_PORT=9400
HTTP_PORT=8090
SCHEDULING="${SCHEDULING:-equal}"

COORD_BIN="target/release/fracture-coordinator-cuda"
WORKER_BIN="target/release/fracture-worker-cuda"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
fail()  { echo -e "${RED}[FAIL]${NC} $*"; exit 1; }

cleanup() {
    info "cleaning up..."
    [ -n "${COORD_PID:-}" ] && kill "$COORD_PID" 2>/dev/null || true
    [ -n "${WORKER_PID:-}" ] && kill "$WORKER_PID" 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup EXIT

# Check prerequisites
[ -f "$COORD_BIN" ] || fail "coordinator binary not found: $COORD_BIN (run: cargo build --release)"
[ -f "$WORKER_BIN" ] || fail "worker binary not found: $WORKER_BIN (run: cargo build --release)"
[ -f "$MODEL_PATH" ] || fail "model not found: $MODEL_PATH"

info "mode: $MODE"
info "model: $MODEL_PATH"
info "scheduling: $SCHEDULING"

if [ "$MODE" = "localhost" ]; then
    # ── Localhost test: coordinator + worker on same machine ──────────
    info "starting coordinator (port $COORD_PORT, http $HTTP_PORT)..."
    RUST_LOG=info "$COORD_BIN" \
        --model "$MODEL_PATH" \
        --listen "127.0.0.1:$COORD_PORT" \
        --workers 1 \
        --http-port "$HTTP_PORT" \
        --scheduling "$SCHEDULING" \
        &
    COORD_PID=$!
    sleep 1

    info "starting worker (connecting to 127.0.0.1:$COORD_PORT)..."
    RUST_LOG=info "$WORKER_BIN" \
        --model "$MODEL_PATH" \
        --coordinator "127.0.0.1:$COORD_PORT" \
        --node-id "worker-local" \
        &
    WORKER_PID=$!

    # Wait for HTTP server to be ready
    info "waiting for HTTP server..."
    for i in $(seq 1 30); do
        if curl -s "http://127.0.0.1:$HTTP_PORT/health" | grep -q "ready"; then
            break
        fi
        [ "$i" -eq 30 ] && fail "HTTP server not ready after 30 seconds"
        sleep 1
    done
    info "HTTP server ready"

elif [ "$MODE" = "cross-machine" ]; then
    # ── Cross-machine test ───────────────────────────────────────────
    [ -n "$WORKER_HOST" ] || fail "usage: $0 cross-machine <worker_host>"

    info "starting coordinator (port $COORD_PORT, http $HTTP_PORT)..."
    RUST_LOG=info "$COORD_BIN" \
        --model "$MODEL_PATH" \
        --listen "0.0.0.0:$COORD_PORT" \
        --workers 1 \
        --http-port "$HTTP_PORT" \
        --scheduling "$SCHEDULING" \
        &
    COORD_PID=$!

    COORD_IP=$(hostname -I | awk '{print $1}')
    info "coordinator IP: $COORD_IP"
    info ""
    info "Start the worker on $WORKER_HOST with:"
    info "  RUST_LOG=info $WORKER_BIN \\"
    info "    --model <model_path> \\"
    info "    --coordinator $COORD_IP:$COORD_PORT"
    info ""
    info "waiting for HTTP server..."
    for i in $(seq 1 120); do
        if curl -s "http://127.0.0.1:$HTTP_PORT/health" | grep -q "ready"; then
            break
        fi
        [ "$i" -eq 120 ] && fail "HTTP server not ready after 120 seconds"
        sleep 1
    done
    info "HTTP server ready"

else
    fail "unknown mode: $MODE (use 'localhost' or 'cross-machine')"
fi

# ── Run inference tests ──────────────────────────────────────────────

info "=== Test 1: Greedy completion ==="
RESPONSE=$(curl -s "http://127.0.0.1:$HTTP_PORT/v1/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "prompt": "The capital of France is",
        "max_tokens": 20,
        "temperature": 0
    }')

echo "$RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$RESPONSE"

# Check response has choices
if echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); assert len(d['choices'])>0; print('  text:', d['choices'][0].get('text','')[:80])" 2>/dev/null; then
    info "Test 1 PASSED"
else
    fail "Test 1 FAILED: no valid completion returned"
fi

info "=== Test 2: Longer generation (50 tokens) ==="
RESPONSE=$(curl -s "http://127.0.0.1:$HTTP_PORT/v1/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "prompt": "Once upon a time",
        "max_tokens": 50,
        "temperature": 0
    }')

COMPLETION_TOKENS=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin)['usage']['completion_tokens'])" 2>/dev/null || echo "0")
if [ "$COMPLETION_TOKENS" -gt 0 ]; then
    info "Test 2 PASSED ($COMPLETION_TOKENS tokens generated)"
else
    fail "Test 2 FAILED: no tokens generated"
fi

info "=== Test 3: Health check ==="
HEALTH=$(curl -s "http://127.0.0.1:$HTTP_PORT/health")
if echo "$HEALTH" | grep -q "ready"; then
    info "Test 3 PASSED"
else
    fail "Test 3 FAILED"
fi

info ""
info "=== All distributed inference tests passed ==="
