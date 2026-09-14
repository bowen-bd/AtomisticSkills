#!/usr/bin/env bash
# Launch one AtomisticSkills MCP server in a container, whichever runtime the
# host has.
#
# Usage: run_server.sh <server-name>
#
# Docker and Apptainer take structurally different arguments -- `docker run
# --rm -i -v h:/work img srv` against `apptainer exec --bind h:/work
# docker://img /entrypoint srv` -- so a single argument list in plugin.json
# cannot serve both. plugin.json therefore invokes this script and passes its
# configuration through the environment.
#
# Apptainer matters here specifically because this software runs on HPC
# clusters, where users have no root and no Docker daemon; Apptainer (formerly
# Singularity) is the standard runtime there.
#
# Configuration (set by plugin.json from userConfig):
#   ATOMISTIC_RUNTIME      docker | podman | apptainer | singularity
#   ATOMISTIC_IMAGE        full image reference, without any transport prefix
#   ATOMISTIC_WORK_DIR     host directory bound to /work
#   ATOMISTIC_MODEL_CACHE  host directory for downloaded model checkpoints
#   ATOMISTIC_GPU          1 to request GPUs, 0 otherwise
#
# stdout is the MCP transport, so every diagnostic here goes to stderr.
set -euo pipefail

SERVER="${1:-}"
RUNTIME="${ATOMISTIC_RUNTIME:-docker}"
IMAGE="${ATOMISTIC_IMAGE:?ATOMISTIC_IMAGE is not set}"
WORK_DIR="${ATOMISTIC_WORK_DIR:-$PWD}"
MODEL_CACHE="${ATOMISTIC_MODEL_CACHE:-$HOME/.cache/atomisticskills}"
WANT_GPU="${ATOMISTIC_GPU:-0}"

die() { printf 'atomisticskills: %s\n' "$*" >&2; exit 1; }

[[ -n "$SERVER" ]] || die "no server name given"

if ! command -v "$RUNTIME" >/dev/null 2>&1; then
    # Claude Code's error sanitiser rewrites a missing binary name to "stdio",
    # producing the baffling 'Executable not found in $PATH: "stdio"'. Say
    # plainly what is missing before the runtime gets a chance to fail.
    available=""
    for c in docker podman apptainer singularity; do
        command -v "$c" >/dev/null 2>&1 && available="$available $c"
    done
    die "container runtime '$RUNTIME' is not on PATH.
       Runtimes found on this host:${available:- none}
       Set the plugin's 'container_runtime' option to one of them:
         claude plugin install atomistic-skills@atomistic-skills --config container_runtime=<name>
       On HPC systems this is normally 'apptainer'."
fi

mkdir -p "$WORK_DIR" "$MODEL_CACHE" 2>/dev/null || true

# Checkpoint caches, shared by both runtimes.
CACHE_ENV=(
    "HF_HOME=/opt/model-cache/huggingface"
    "TORCH_HOME=/opt/model-cache/torch"
    "MATGL_CACHE=/opt/model-cache/matgl"
)

case "$RUNTIME" in
    docker|podman)
        args=(run --rm --interactive
              --volume "${WORK_DIR}:/work" --workdir /work
              --volume "${MODEL_CACHE}:/opt/model-cache")
        for kv in "${CACHE_ENV[@]}"; do args+=(--env "$kv"); done
        # Forward credentials only when the host actually has them set, so an
        # unset variable does not become an empty one inside the container.
        for v in MP_API_KEY HF_TOKEN OPENALEX_EMAIL ELSEVIER_API_KEY \
                 ELSEVIER_INST_TOKEN SPRINGER_API_KEY UNPAYWALL_EMAIL; do
            [[ -n "${!v:-}" ]] && args+=(--env "$v=${!v}")
        done
        [[ "$WANT_GPU" == "1" ]] && args+=(--gpus all)
        args+=("$IMAGE" "$SERVER")
        exec "$RUNTIME" "${args[@]}"
        ;;

    apptainer|singularity)
        # Apptainer converts the OCI image to a SIF on first use. Keep that
        # conversion off the home quota, which is small on most clusters.
        export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-${MODEL_CACHE}/apptainer-cache}"
        export SINGULARITY_CACHEDIR="${SINGULARITY_CACHEDIR:-$APPTAINER_CACHEDIR}"
        mkdir -p "$APPTAINER_CACHEDIR" 2>/dev/null || true

        args=(exec
              --bind "${WORK_DIR}:/work"
              --bind "${MODEL_CACHE}:/opt/model-cache"
              --pwd /work)
        for kv in "${CACHE_ENV[@]}"; do args+=(--env "$kv"); done
        for v in MP_API_KEY HF_TOKEN OPENALEX_EMAIL ELSEVIER_API_KEY \
                 ELSEVIER_INST_TOKEN SPRINGER_API_KEY UNPAYWALL_EMAIL; do
            [[ -n "${!v:-}" ]] && args+=(--env "$v=${!v}")
        done
        [[ "$WANT_GPU" == "1" ]] && args+=(--nv)
        # `exec` bypasses the image ENTRYPOINT, so name it explicitly. Apptainer
        # already runs as the invoking user, so the entrypoint's privilege drop
        # is a no-op there.
        args+=("docker://${IMAGE}" /opt/atomisticskills/docker/entrypoint.sh "$SERVER")
        exec "$RUNTIME" "${args[@]}"
        ;;

    *)
        die "unsupported container_runtime '$RUNTIME' (expected docker, podman, apptainer or singularity)"
        ;;
esac
