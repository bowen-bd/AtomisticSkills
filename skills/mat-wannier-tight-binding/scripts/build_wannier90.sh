#!/usr/bin/env bash
# Build a pinned Wannier90 release and install its executables and library.
# Usage: WANNIER_PREFIX=/path/to/install bash build_wannier90.sh
# Requires git, CMake >= 3.25, a Fortran compiler, BLAS and LAPACK.
# Overrides: WANNIER_REF, WANNIER_BUILD_DIR, BUILD_JOBS, FC,
#            BLAS_LIBRARIES, LAPACK_LIBRARIES.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
WANNIER_REF="${WANNIER_REF:-v4.0.3}"
WANNIER_BUILD_DIR="${WANNIER_BUILD_DIR:-${REPO_ROOT}/.agents/test/wannier90-build}"
WANNIER_PREFIX="${WANNIER_PREFIX:-${WANNIER_BUILD_DIR}/install}"
BUILD_JOBS="${BUILD_JOBS:-4}"

for cmd in git cmake; do
    command -v "$cmd" >/dev/null || { echo "Required command missing: $cmd" >&2; exit 1; }
done
[[ "$WANNIER_BUILD_DIR" = /* && "$WANNIER_PREFIX" = /* ]] || {
    echo 'WANNIER_BUILD_DIR and WANNIER_PREFIX must be absolute paths.' >&2; exit 1;
}
[[ "$BUILD_JOBS" =~ ^[1-9][0-9]*$ ]] || { echo 'BUILD_JOBS must be positive.' >&2; exit 1; }

mkdir -p "$WANNIER_BUILD_DIR"
if [[ ! -d "$WANNIER_BUILD_DIR/src/.git" ]]; then
    git clone --depth 1 --branch "$WANNIER_REF" \
        https://github.com/wannier-developers/wannier90.git "$WANNIER_BUILD_DIR/src"
fi
requested_commit="$(git -C "$WANNIER_BUILD_DIR/src" rev-parse "${WANNIER_REF}^{commit}")"
actual_commit="$(git -C "$WANNIER_BUILD_DIR/src" rev-parse HEAD)"
if [[ "$requested_commit" != "$actual_commit" ]] || \
   [[ -n "$(git -C "$WANNIER_BUILD_DIR/src" status --porcelain --untracked-files=no)" ]]; then
    echo 'Existing source differs from the requested release; use a fresh WANNIER_BUILD_DIR.' >&2
    exit 1
fi

cmake_args=(
    -S "$WANNIER_BUILD_DIR/src" -B "$WANNIER_BUILD_DIR/build"
    "-DCMAKE_INSTALL_PREFIX=$WANNIER_PREFIX" -DCMAKE_BUILD_TYPE=Release
    -DWANNIER90_SHARED_LIBS=OFF -DWANNIER90_TEST=OFF -DWANNIER90_INSTALL=ON
)
[[ -z "${FC:-}" ]] || cmake_args+=("-DCMAKE_Fortran_COMPILER=$FC")
# Runtime-only BLAS/LAPACK packages may lack unversioned .so symlinks.
for library in BLAS LAPACK; do
    variable="${library}_LIBRARIES"
    value="${!variable:-}"
    if [[ -z "$value" ]] && command -v ldconfig >/dev/null; then
        soname="lib${library,,}.so.3"
        value="$(ldconfig -p | awk -v name="$soname" '$1 == name && !found {value=$NF; found=1} END {print value}')"
    fi
    [[ -z "$value" ]] || cmake_args+=("-D${variable}=$value")
done

cmake "${cmake_args[@]}"
cmake --build "$WANNIER_BUILD_DIR/build" --parallel "$BUILD_JOBS"
cmake --install "$WANNIER_BUILD_DIR/build"
# Execute directly: command substitution inside echo would hide a failed probe.
"$WANNIER_PREFIX/bin/wannier90.x" -v
printf 'Wannier90 %s (%s) installed in %s\n' "$WANNIER_REF" "$actual_commit" "$WANNIER_PREFIX"
printf 'Add %s/bin to PATH to use this installation.\n' "$WANNIER_PREFIX"
