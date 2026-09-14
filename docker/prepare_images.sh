#!/usr/bin/env bash
# Pre-build the container images this host can run, before Claude Code needs them.
#
# Usage:
#   bash docker/prepare_images.sh [--runtime apptainer] [--registry ghcr.io/...]
#                                 [--tag VERSION] [--cache DIR] [image ...]
#
# Why this exists: Claude Code probes every MCP server in parallel and gives
# each 30 seconds to connect. Converting a 3.5 GB OCI image to a SIF takes
# minutes, so on a first run under Apptainer every server times out, and the
# parallel probe starts several conversions of the same image at once -- one
# HPC test consumed 29 GB of quota that way. Running this first turns that into
# a single, sequential, visible step.
#
# Only images matching this machine's architecture are prepared; the rest are
# reported and skipped rather than downloaded.
#
# Requirements:
#   - apptainer/singularity (SIF cache) or docker/podman (image pull)
#   - jq is not required; image metadata is read with python3
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPEC="${HERE}/images.json"

RUNTIME=""
REGISTRY="ghcr.io/learningmatter-mit"
TAG=""
CACHE=""
WANTED=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --runtime)  RUNTIME="$2"; shift 2 ;;
        --registry) REGISTRY="$2"; shift 2 ;;
        --tag)      TAG="$2"; shift 2 ;;
        --cache)    CACHE="$2"; shift 2 ;;
        -h|--help)  sed -n '2,20p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *)          WANTED+=("$1"); shift ;;
    esac
done

log() { printf '%s\n' "$*" >&2; }
die() { log "prepare_images: $*"; exit 1; }

[[ -f "$SPEC" ]] || die "cannot find ${SPEC}"
[[ -n "$TAG" ]] || TAG="$(cat "${HERE}/../VERSION" 2>/dev/null || echo latest)"

if [[ -z "$RUNTIME" ]]; then
    for c in apptainer singularity docker podman; do
        command -v "$c" >/dev/null 2>&1 && { RUNTIME="$c"; break; }
    done
fi
[[ -n "$RUNTIME" ]] || die "no container runtime found (looked for apptainer, singularity, docker, podman)"
command -v "$RUNTIME" >/dev/null 2>&1 || die "runtime '$RUNTIME' is not on PATH"

case "$(uname -m)" in
    x86_64|amd64)  HOST_ARCH=amd64 ;;
    aarch64|arm64) HOST_ARCH=arm64 ;;
    *)             HOST_ARCH="$(uname -m)" ;;
esac

# Where to put the SIF. run_server.sh searches both this script's possible
# choices, so the pre-build is found either way -- do not "fix" this by trying
# harder to guess CLAUDE_PLUGIN_DATA, which is the mistake this replaced.
#
# The plugin's data directory is preferred when it exists, so the SIF sits
# beside the model cache the servers use at runtime. It usually does NOT exist
# yet: Claude Code creates CLAUDE_PLUGIN_DATA on the first session that loads
# the plugin, and this script is meant to run before that session. The shared
# fallback is then correct, not a degraded mode.
if [[ -z "$CACHE" ]]; then
    if [[ -n "${ATOMISTIC_MODEL_CACHE:-}" ]]; then
        CACHE="$ATOMISTIC_MODEL_CACHE"
    else
        plugin_data=""
        for d in "$HOME"/.claude/plugins/data/*atomistic-skills*/; do
            [[ -d "$d" ]] && plugin_data="${d%/}"
        done
        if [[ -n "$plugin_data" ]]; then
            CACHE="${plugin_data}/model-cache"
            log "using the plugin's data directory: ${CACHE}"
        else
            CACHE="$HOME/.cache/atomisticskills"
            log "the plugin's data directory does not exist yet (Claude Code"
            log "creates it on the first session); using the shared cache:"
            log "  ${CACHE}"
            log "run_server.sh looks here too, so this pre-build will be used."
        fi
    fi
fi

log "runtime:  $RUNTIME"
log "arch:     linux/${HOST_ARCH}"
log "registry: ${REGISTRY}"
log "tag:      ${TAG}"
log "cache:    ${CACHE}"
log ""

# name<TAB>platforms, straight from the single source of truth.
mapfile -t rows < <(python3 -c "
import json, sys
spec = json.load(open('${SPEC}'))
for i in spec['images']:
    print(i['name'] + '\t' + ','.join(i['platforms']))
")

prepared=0 skipped=0 failed=0
for row in "${rows[@]}"; do
    name="${row%%$'\t'*}"
    platforms="${row##*$'\t'}"

    if [[ ${#WANTED[@]} -gt 0 ]] && [[ " ${WANTED[*]} " != *" ${name} "* ]]; then
        continue
    fi
    if [[ ",${platforms}," != *",linux/${HOST_ARCH},"* ]]; then
        log "skip  ${name}  (built for ${platforms//,/ }, not linux/${HOST_ARCH})"
        skipped=$((skipped + 1))
        continue
    fi

    image="${REGISTRY}/atomisticskills-${name}:${TAG}"

    case "$RUNTIME" in
        apptainer|singularity)
            export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-${CACHE}/apptainer-cache}"
            export SINGULARITY_CACHEDIR="${SINGULARITY_CACHEDIR:-$APPTAINER_CACHEDIR}"
            export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-${CACHE}/apptainer-tmp}"
            export SINGULARITY_TMPDIR="${SINGULARITY_TMPDIR:-$APPTAINER_TMPDIR}"
            # See run_server.sh: mksquashfs spawns one thread per core and dies
            # against a low `ulimit -u` on large HPC nodes.
            if [[ -z "${ATOMISTIC_SQUASHFS_PROCS:-}" ]]; then
                nproc_count="$(nproc 2>/dev/null || echo 4)"
                proc_limit="$(ulimit -u 2>/dev/null || echo 4096)"
                [[ "$proc_limit" == "unlimited" ]] && proc_limit=4096
                headroom=$(( (proc_limit - 256) / 8 )); (( headroom < 1 )) && headroom=1
                ATOMISTIC_SQUASHFS_PROCS=$(( nproc_count < headroom ? nproc_count : headroom ))
                (( ATOMISTIC_SQUASHFS_PROCS > 8 )) && ATOMISTIC_SQUASHFS_PROCS=8
            fi
            export APPTAINER_MKSQUASHFS_ARGS="${APPTAINER_MKSQUASHFS_ARGS:--processors ${ATOMISTIC_SQUASHFS_PROCS}}"
            export SINGULARITY_MKSQUASHFS_ARGS="$APPTAINER_MKSQUASHFS_ARGS"

            mkdir -p "${CACHE}/sif" "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
            sif="${CACHE}/sif/atomisticskills-${name}-${TAG}.sif"
            if [[ -f "$sif" ]]; then
                log "have  ${name}  ($(du -h "$sif" | cut -f1))"
                prepared=$((prepared + 1))
                continue
            fi
            log "build ${name}  <- ${image}  (mksquashfs -processors ${ATOMISTIC_SQUASHFS_PROCS})"
            if "$RUNTIME" build --force "${sif}.partial" "docker://${image}" >&2; then
                mv -f "${sif}.partial" "$sif"
                log "  ok  $(du -h "$sif" | cut -f1)"
                prepared=$((prepared + 1))
            else
                rm -f "${sif}.partial"
                log "  FAILED ${name}"
                failed=$((failed + 1))
            fi
            ;;
        docker|podman)
            log "pull  ${name}  <- ${image}"
            if "$RUNTIME" pull "$image" >&2; then
                prepared=$((prepared + 1))
            else
                log "  FAILED ${name}"
                failed=$((failed + 1))
            fi
            ;;
        *)
            die "unsupported runtime '$RUNTIME'"
            ;;
    esac
done

log ""
log "prepared ${prepared}, skipped ${skipped} (wrong architecture), failed ${failed}"
[[ $failed -eq 0 ]]
