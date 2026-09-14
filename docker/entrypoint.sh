#!/usr/bin/env bash
# Launch one AtomisticSkills MCP server over stdio.
#
# Usage: docker run -i --rm <image> <server-name>
#
# The server name is resolved against /opt/atomisticskills/docker/server-map.txt,
# which is generated at build time by `docker/render.py server-map <image>`.
# Resolution is done in pure shell because it has to happen before any conda
# environment is activated, so no interpreter is assumed to be on PATH.
#
# stdout is the MCP transport and must carry nothing but protocol frames, so
# every diagnostic here goes to stderr.
set -euo pipefail

REPO_DIR="${ATOMISTIC_REPO_DIR:-/opt/atomisticskills}"
MAP_FILE="${REPO_DIR}/docker/server-map.txt"

die() { printf '%s\n' "$*" >&2; exit 1; }

available() { grep -v '^#' "$MAP_FILE" | cut -d: -f1 | paste -sd' ' -; }

[[ -f "$MAP_FILE" ]] || die "server map missing at ${MAP_FILE}; image built incorrectly"

SERVER="${1:-}"
if [[ -z "$SERVER" ]]; then
    die "usage: docker run -i --rm <image> <server>
servers in this image: $(available)"
fi

# Skills write results into the bind-mounted /work. Running as root would leave
# root-owned files in the caller's project, so match /work's owner instead.
# Re-exec once (guarded by the marker) and fall back to root if setpriv is
# unavailable -- a readable result tree matters more than the ownership nicety.
if [[ -z "${ATOMISTIC_UID_MATCHED:-}" && "$(id -u)" == "0" ]] && [[ -d /work ]]; then
    WORK_UID="$(stat -c %u /work 2>/dev/null || echo 0)"
    WORK_GID="$(stat -c %g /work 2>/dev/null || echo 0)"
    if [[ "$WORK_UID" != "0" ]] && command -v setpriv >/dev/null 2>&1; then
        export ATOMISTIC_UID_MATCHED=1 HOME=/tmp
        chown "$WORK_UID:$WORK_GID" /opt/model-cache 2>/dev/null || true
        exec setpriv --reuid "$WORK_UID" --regid "$WORK_GID" --clear-groups \
            "$0" "$@"
    fi
fi

ROW="$(grep -v '^#' "$MAP_FILE" | grep "^${SERVER}:" || true)"
[[ -n "$ROW" ]] || die "unknown server '${SERVER}' in this image
servers in this image: $(available)"

ENV_NAME="$(cut -d: -f2 <<<"$ROW")"
MODULE="$(cut -d: -f3 <<<"$ROW")"
EXTRA_ENV="$(cut -d: -f4 <<<"$ROW")"

# Per-server environment variables, e.g. MATGL_BACKEND=DGL.
if [[ -n "$EXTRA_ENV" ]]; then
    while IFS= read -r pair; do
        [[ -n "$pair" ]] && export "${pair?}"
    done < <(tr ',' '\n' <<<"$EXTRA_ENV")
fi

# The servers are imported as `src.mcp_server.*`, so the repo root has to be
# importable regardless of the working directory the caller mounted.
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1

# Keep multi-GB checkpoint downloads on the mounted cache volume rather than in
# the container's writable layer, where they would be lost on exit.
export HF_HOME="${HF_HOME:-/opt/model-cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-/opt/model-cache/torch}"
mkdir -p "$HF_HOME" "$TORCH_HOME" 2>/dev/null || true

printf 'atomisticskills: starting %s (env=%s) \n' "$SERVER" "$ENV_NAME" >&2

# Run from the mounted workspace, not the repository. Under Apptainer the
# repository is a read-only SquashFS, so a tool given a relative output path
# fails with "[Errno 30] Read-only file system". PYTHONPATH above already makes
# the repository importable, so nothing depends on it being the cwd.
WORKSPACE=/work
if [[ -d "$WORKSPACE" && -w "$WORKSPACE" ]]; then
    cd "$WORKSPACE"
else
    # No writable mount (a bare `docker run` with no -v): fall back to the
    # repository, which is writable under Docker even if not under Apptainer.
    WORKSPACE="$REPO_DIR"
    cd "$REPO_DIR"
    printf 'atomisticskills: /work is not writable; using %s\n' "$REPO_DIR" >&2
fi
# Tells src/utils/research_utils.py where research/ and .env belong.
export ATOMISTIC_WORKSPACE="${ATOMISTIC_WORKSPACE:-$WORKSPACE}"

exec micromamba run -n "$ENV_NAME" python -m "$MODULE" "${@:2}"
