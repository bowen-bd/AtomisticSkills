#!/usr/bin/env bash
# Entrypoint of every AtomisticSkills image.
#
#   docker run -i --rm <image> <server>        start an MCP server over stdio
#   docker run --rm <image> <command> [args]   run a command in the environment
#
# venv/run is the normal caller: it mounts the host's workspace and repository at
# their own paths and passes ATOMISTIC_WORKSPACE and PYTHONPATH, so a container
# reads and writes exactly the paths the host does.
#
# The server name is resolved against docker/server-map.txt, generated at build
# time by `docker/render.py server-map <image>`. Each server runs in its uv
# environment, venv/<venv>/.venv; the generative image carries several.
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
[[ -n "$SERVER" ]] || die "usage: docker run -i --rm <image> <server> | <command> [args...]
servers in this image: $(available)"

# Started by hand as root with a project bound to /work (the 1.4 layout): match
# /work's owner so results are not left owned by root. venv/run passes --user
# instead, so this only runs for direct `docker run` use.
if [[ -z "${ATOMISTIC_UID_MATCHED:-}" && "$(id -u)" == "0" && -d /work ]]; then
    WORK_UID="$(stat -c %u /work 2>/dev/null || echo 0)"
    WORK_GID="$(stat -c %g /work 2>/dev/null || echo 0)"
    if [[ "$WORK_UID" != "0" ]] && command -v setpriv >/dev/null 2>&1; then
        export ATOMISTIC_UID_MATCHED=1 HOME=/tmp
        chown "$WORK_UID:$WORK_GID" /opt/model-cache 2>/dev/null || true
        exec setpriv --reuid "$WORK_UID" --regid "$WORK_GID" --clear-groups "$0" "$@"
    fi
fi

# A command runs in one environment: the one ATOMISTIC_VENV names (venv/run
# passes it), else the image's first. An image may carry several (generative).
VENV_NAME="${ATOMISTIC_VENV:-}"
if [[ -z "$VENV_NAME" ]]; then
    for d in "$REPO_DIR"/venv/*/.venv; do
        [[ -d "$d" ]] && { VENV_NAME="$(basename "$(dirname "$d")")"; break; }
    done
fi
if [[ -n "$VENV_NAME" && -x "${REPO_DIR}/venv/${VENV_NAME}/.venv/bin/python" ]]; then
    export VIRTUAL_ENV="${REPO_DIR}/venv/${VENV_NAME}/.venv"
    export PATH="${VIRTUAL_ENV}/bin:${PATH}"
fi

ROW="$(grep -v '^#' "$MAP_FILE" | grep "^${SERVER}:" || true)"
if [[ -z "$ROW" ]]; then
    # Not a server name: run it as a command. That is how venv/run executes a
    # skill script, and how an image is inspected (`docker run IMG bash`). Say
    # so, so a mistyped server name can never silently become another command.
    command -v "$SERVER" >/dev/null 2>&1 \
        || die "unknown server or command '${SERVER}'
servers in this image: $(available)"
    exec "$@"
fi

ENV_NAME="$(cut -d: -f2 <<<"$ROW")"
MODULE="$(cut -d: -f3 <<<"$ROW")"
EXTRA_ENV="$(cut -d: -f4 <<<"$ROW")"
if [[ -n "$EXTRA_ENV" ]]; then
    while IFS= read -r pair; do
        [[ -n "$pair" ]] && export "${pair?}"
    done < <(tr ',' '\n' <<<"$EXTRA_ENV")
fi

# venv/run passes the host repository as PYTHONPATH so the container runs the
# host's code; otherwise use the image's own copy.
export PYTHONPATH="${PYTHONPATH:-$REPO_DIR}"
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/opt/model-cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-/opt/model-cache/torch}"
mkdir -p "$HF_HOME" "$TORCH_HOME" 2>/dev/null || true

# Run from the workspace, never from the repository: under Apptainer the
# repository is a read-only SquashFS, and a relative output path would fail.
WORKSPACE="${ATOMISTIC_WORKSPACE:-}"
if [[ -z "$WORKSPACE" || ! -d "$WORKSPACE" ]]; then
    if [[ -d /work && -w /work ]]; then WORKSPACE=/work; else WORKSPACE="$REPO_DIR"; fi
fi
cd "$WORKSPACE"
export ATOMISTIC_WORKSPACE="$WORKSPACE"

printf 'atomisticskills: starting %s (env=%s, workspace=%s)\n' "$SERVER" "$ENV_NAME" "$WORKSPACE" >&2

[[ -x "${REPO_DIR}/venv/${ENV_NAME}/.venv/bin/python" ]] \
    || die "environment ${ENV_NAME} for server ${SERVER} missing from this image"
exec "${REPO_DIR}/venv/${ENV_NAME}/.venv/bin/python" -m "$MODULE" "${@:2}"
