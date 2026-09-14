#!/usr/bin/env bash
# Launch one AtomisticSkills MCP server in a container, whichever runtime the
# host has.
#
# Usage: run_server.sh <server-name>
#
# Docker and Apptainer take structurally different arguments -- `docker run
# --rm -i -v h:/work img srv` against `apptainer exec --bind h:/work img
# /entrypoint srv` -- so a single argument list in plugin.json cannot serve
# both. plugin.json therefore invokes this script and passes its configuration
# through the environment.
#
# Apptainer matters here specifically because this software runs on HPC
# clusters, where users have no root and no Docker daemon; Apptainer (formerly
# Singularity) is the standard runtime there.
#
# Configuration (set by plugin.json from userConfig):
#   ATOMISTIC_RUNTIME      docker | podman | apptainer | singularity
#   ATOMISTIC_IMAGE        full image reference, without any transport prefix
#   ATOMISTIC_IMAGE_NAME   short image name, used for the SIF filename
#   ATOMISTIC_PLATFORMS    comma-separated platforms the image was built for
#   ATOMISTIC_WORK_DIR     host directory bound to /work
#   ATOMISTIC_MODEL_CACHE  host directory for checkpoints and cached SIFs
#   ATOMISTIC_GPU          1 to request GPUs, 0 otherwise
#
# Optional overrides:
#   ATOMISTIC_SQUASHFS_PROCS  mksquashfs worker count (default: bounded, see below)
#   ATOMISTIC_BUILD_TIMEOUT   seconds to wait for another process's SIF build
#
# stdout is the MCP transport, so every diagnostic here goes to stderr.
set -euo pipefail

SERVER="${1:-}"
RUNTIME="${ATOMISTIC_RUNTIME:-docker}"
IMAGE="${ATOMISTIC_IMAGE:-}"
IMAGE_NAME="${ATOMISTIC_IMAGE_NAME:-image}"
PLATFORMS="${ATOMISTIC_PLATFORMS:-}"
WORK_DIR="${ATOMISTIC_WORK_DIR:-$PWD}"
MODEL_CACHE="${ATOMISTIC_MODEL_CACHE:-$HOME/.cache/atomisticskills}"
WANT_GPU="${ATOMISTIC_GPU:-0}"

log() { printf 'atomisticskills: %s\n' "$*" >&2; }
die() { log "$*"; exit 1; }

[[ -n "$SERVER" ]] || die "no server name given"

# --- architecture gate -------------------------------------------------------
# The GPU images are built for arm64 only. Without this check, an x86_64 host
# quietly downloads and unpacks them anyway -- one HPC test burned 29 GB of
# quota pulling four images in parallel before every server failed. Refuse
# early and say why.
case "$(uname -m)" in
    x86_64|amd64) HOST_ARCH=amd64 ;;
    aarch64|arm64) HOST_ARCH=arm64 ;;
    *) HOST_ARCH="$(uname -m)" ;;
esac

if [[ -n "$PLATFORMS" ]] && [[ ",${PLATFORMS}," != *",linux/${HOST_ARCH},"* ]]; then
    die "server '${SERVER}' is unavailable on this machine.
       Its image (${IMAGE_NAME}) is built for: ${PLATFORMS//,/ }
       This host is linux/${HOST_ARCH}.
       Nothing is downloaded. The other servers in this plugin still work;
       see docker/README.md for which images cover which architectures."
fi

[[ -n "$IMAGE" ]] || die "ATOMISTIC_IMAGE is not set"

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

CACHE_ENV=(
    "HF_HOME=/opt/model-cache/huggingface"
    "TORCH_HOME=/opt/model-cache/torch"
    "MATGL_CACHE=/opt/model-cache/matgl"
)
# Credentials the servers need, forwarded only when the host actually has them
# set, so an unset variable does not arrive as an empty one.
CREDENTIALS=(MP_API_KEY HF_TOKEN OPENALEX_EMAIL ELSEVIER_API_KEY
             ELSEVIER_INST_TOKEN SPRINGER_API_KEY UNPAYWALL_EMAIL)

case "$RUNTIME" in
    docker|podman)
        args=(run --rm --interactive
              --volume "${WORK_DIR}:/work" --workdir /work
              --volume "${MODEL_CACHE}:/opt/model-cache")
        for kv in "${CACHE_ENV[@]}"; do args+=(--env "$kv"); done
        for v in "${CREDENTIALS[@]}"; do
            [[ -n "${!v:-}" ]] && args+=(--env "$v=${!v}")
        done
        [[ "$WANT_GPU" == "1" ]] && args+=(--gpus all)
        args+=("$IMAGE" "$SERVER")
        exec "$RUNTIME" "${args[@]}"
        ;;

    apptainer|singularity)
        # Build once into a cached SIF and exec that, rather than resolving
        # docker:// on every launch. Claude Code probes all servers in parallel
        # and gives each 30s; an on-the-fly conversion of a 3.5 GB image takes
        # far longer than that and, run concurrently, converts the same image
        # several times over.
        export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-${MODEL_CACHE}/apptainer-cache}"
        export SINGULARITY_CACHEDIR="${SINGULARITY_CACHEDIR:-$APPTAINER_CACHEDIR}"
        # Clusters often mount /tmp with nodev, which Apptainer warns can break
        # the build. Keep temporary files beside the cache instead.
        export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-${MODEL_CACHE}/apptainer-tmp}"
        export SINGULARITY_TMPDIR="${SINGULARITY_TMPDIR:-$APPTAINER_TMPDIR}"

        # mksquashfs defaults to one thread per core. On a 448-core login node
        # with `ulimit -u` of 768 that exhausts the thread limit outright:
        # "FATAL ERROR: Failed to create thread". Bound it well under whatever
        # headroom the user actually has.
        if [[ -z "${ATOMISTIC_SQUASHFS_PROCS:-}" ]]; then
            nproc_count="$(nproc 2>/dev/null || echo 4)"
            proc_limit="$(ulimit -u 2>/dev/null || echo 4096)"
            [[ "$proc_limit" == "unlimited" ]] && proc_limit=4096
            headroom=$(( (proc_limit - 256) / 8 ))
            (( headroom < 1 )) && headroom=1
            ATOMISTIC_SQUASHFS_PROCS=$(( nproc_count < headroom ? nproc_count : headroom ))
            (( ATOMISTIC_SQUASHFS_PROCS > 8 )) && ATOMISTIC_SQUASHFS_PROCS=8
        fi
        export APPTAINER_MKSQUASHFS_ARGS="${APPTAINER_MKSQUASHFS_ARGS:--processors ${ATOMISTIC_SQUASHFS_PROCS}}"
        export SINGULARITY_MKSQUASHFS_ARGS="$APPTAINER_MKSQUASHFS_ARGS"

        mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" "${MODEL_CACHE}/sif" 2>/dev/null || true

        # Find a pre-built SIF wherever prepare_images.sh actually put it.
        #
        # That script cannot reliably predict this directory. plugin.json points
        # ATOMISTIC_MODEL_CACHE at ${CLAUDE_PLUGIN_DATA}/model-cache, but Claude
        # Code does not create CLAUDE_PLUGIN_DATA until the first session loads
        # the plugin -- which is *after* the documented pre-build step. So the
        # pre-build legitimately lands in the shared fallback, and an HPC test
        # then sat through four 30s connect timeouts with a perfectly good SIF
        # on disk. Searching from this side is what makes the two agree;
        # predicting from the other side cannot.
        sif_name="atomisticskills-${IMAGE_NAME}-${IMAGE##*:}.sif"
        sif="${MODEL_CACHE}/sif/${sif_name}"
        if [[ ! -f "$sif" ]]; then
            for alt_dir in "${ATOMISTIC_SIF_DIR:-}" "$HOME/.cache/atomisticskills"; do
                [[ -n "$alt_dir" && -f "${alt_dir}/sif/${sif_name}" ]] || continue
                sif="${alt_dir}/sif/${sif_name}"
                log "using pre-built SIF from ${alt_dir}/sif"
                break
            done
        fi

        if [[ ! -f "$sif" ]]; then
            # Serialise: ten servers starting at once must not each build the
            # same image. Whoever gets the lock builds; the rest wait and reuse.
            lock="${MODEL_CACHE}/sif/.${IMAGE_NAME}.lock"
            build_timeout="${ATOMISTIC_BUILD_TIMEOUT:-1800}"
            log "no cached SIF for ${IMAGE_NAME}; building (this takes minutes -- run docker/prepare_images.sh beforehand to avoid connection timeouts)"
            if command -v flock >/dev/null 2>&1; then
                exec {lockfd}>"$lock"
                flock -w "$build_timeout" "$lockfd" \
                    || die "timed out waiting ${build_timeout}s for another process to build ${IMAGE_NAME}"
            fi
            if [[ ! -f "$sif" ]]; then
                tmp_sif="${sif}.$$.partial"
                "$RUNTIME" build --force "$tmp_sif" "docker://${IMAGE}" >&2 \
                    || die "failed to build SIF for ${IMAGE}.
       If this reported 'Failed to create thread', mksquashfs exceeded the
       thread limit; retry with ATOMISTIC_SQUASHFS_PROCS=2."
                mv -f "$tmp_sif" "$sif"
            fi
            [[ -n "${lockfd:-}" ]] && exec {lockfd}>&-
        fi

        args=(exec
              --bind "${WORK_DIR}:/work"
              --bind "${MODEL_CACHE}:/opt/model-cache"
              --pwd /work)
        for kv in "${CACHE_ENV[@]}"; do args+=(--env "$kv"); done
        for v in "${CREDENTIALS[@]}"; do
            [[ -n "${!v:-}" ]] && args+=(--env "$v=${!v}")
        done
        [[ "$WANT_GPU" == "1" ]] && args+=(--nv)
        # `exec` bypasses the image ENTRYPOINT, so name it explicitly. Apptainer
        # already runs as the invoking user, so the entrypoint's privilege drop
        # is a no-op there.
        args+=("$sif" /opt/atomisticskills/docker/entrypoint.sh "$SERVER")
        exec "$RUNTIME" "${args[@]}"
        ;;

    *)
        die "unsupported container_runtime '$RUNTIME' (expected docker, podman, apptainer or singularity)"
        ;;
esac
