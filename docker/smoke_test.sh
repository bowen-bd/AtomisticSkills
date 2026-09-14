#!/usr/bin/env bash
# Verify that a built image actually speaks MCP, not merely that it built.
#
# Drives a real stdio handshake against the container -- initialize, then
# tools/list -- and fails unless the server returns its identity and a
# non-empty tool list. An image that builds green but whose entrypoint cannot
# start a server is precisely the silent breakage these images exist to
# prevent, so this runs in CI on every push.
#
# Usage:
#   bash docker/smoke_test.sh <image-ref> <server-name> [more servers...]
#
# Requirements:
#   - docker (or set RUNTIME=podman)
#   - python3 on PATH for response parsing
set -uo pipefail

RUNTIME="${RUNTIME:-docker}"
TIMEOUT="${TIMEOUT:-180}"

IMAGE="${1:-}"
shift || true
SERVERS=("$@")

if [[ -z "$IMAGE" || ${#SERVERS[@]} -eq 0 ]]; then
    echo "usage: bash docker/smoke_test.sh <image-ref> <server> [server...]" >&2
    exit 2
fi

PROTOCOL_VERSION="${PROTOCOL_VERSION:-2024-11-05}"

# stdin must stay open until the replies arrive: an MCP server reads EOF as a
# shutdown and can exit before flushing. A fixed sleep is the wrong instrument
# for that -- 8s sufficed locally but not for atomate2 on a cold CI runner,
# whose imports are slow, so a healthy image failed on timing alone. Hold the
# pipe open through a FIFO instead and close it as soon as the responses land,
# falling back to REPLY_DEADLINE only if they never do.
REPLY_DEADLINE="${REPLY_DEADLINE:-120}"

failures=0
for server in "${SERVERS[@]}"; do
    printf '== %s :: %s\n' "$IMAGE" "$server"

    workdir="$(mktemp -d)"
    fifo="$workdir/stdin"
    mkfifo "$fifo"

    timeout "$TIMEOUT" "$RUNTIME" run --rm --interactive \
        --volume "$PWD:/work" --workdir /work "$IMAGE" "$server" \
        < "$fifo" > "$workdir/out" 2>"$workdir/err" &
    runner=$!

    # Hold the write end open for the whole exchange.
    exec {req}> "$fifo"
    cat >&${req} <<EOF
{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"${PROTOCOL_VERSION}","capabilities":{},"clientInfo":{"name":"atomisticskills-smoke","version":"1"}}}
{"jsonrpc":"2.0","method":"notifications/initialized","params":{}}
{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}
EOF

    # Close as soon as the tools/list reply appears, rather than guessing.
    waited=0
    while [[ $waited -lt $REPLY_DEADLINE ]]; do
        grep -q '"id":[[:space:]]*2' "$workdir/out" 2>/dev/null && break
        kill -0 "$runner" 2>/dev/null || break
        sleep 1
        waited=$((waited + 1))
    done
    [[ $waited -ge $REPLY_DEADLINE ]] && echo "  (no reply after ${REPLY_DEADLINE}s)" >&2

    exec {req}>&-
    wait "$runner" 2>/dev/null
    rc=$?

    cp "$workdir/out" /tmp/smoke.out
    cp "$workdir/err" /tmp/smoke.err
    rm -rf "$workdir"

    # A server that exits non-zero after answering is still a pass: it is
    # reacting to stdin closing, not to a protocol error.
    if ! python3 - "$server" /tmp/smoke.out <<'PY'
import json, sys

server, path = sys.argv[1], sys.argv[2]
frames = []
for line in open(path, encoding="utf-8", errors="replace"):
    line = line.strip()
    if not line.startswith("{"):
        continue
    try:
        frames.append(json.loads(line))
    except json.JSONDecodeError:
        pass

init = next((f for f in frames if f.get("id") == 1), None)
tools = next((f for f in frames if f.get("id") == 2), None)

if init is None:
    print(f"  FAIL: no response to initialize"); sys.exit(1)
if "error" in init:
    print(f"  FAIL: initialize returned error: {init['error']}"); sys.exit(1)

info = init.get("result", {}).get("serverInfo", {})
print(f"  initialize OK -> {info.get('name','?')} {info.get('version','?')}")

if tools is None:
    print("  FAIL: no response to tools/list"); sys.exit(1)
if "error" in tools:
    print(f"  FAIL: tools/list returned error: {tools['error']}"); sys.exit(1)

names = [t.get("name") for t in tools.get("result", {}).get("tools", [])]
if not names:
    print("  FAIL: server exposed zero tools"); sys.exit(1)
print(f"  tools/list OK -> {len(names)} tools: {', '.join(names[:6])}"
      + (" ..." if len(names) > 6 else ""))
PY
    then
        failures=$((failures + 1))
        echo "  --- container stderr (last 20 lines) ---" >&2
        tail -20 /tmp/smoke.err >&2 || true
        echo "  (runtime exit code was ${rc})" >&2
    fi
done

if [[ $failures -gt 0 ]]; then
    echo "smoke test FAILED for ${failures} server(s)" >&2
    exit 1
fi
echo "smoke test passed for ${#SERVERS[@]} server(s)"
