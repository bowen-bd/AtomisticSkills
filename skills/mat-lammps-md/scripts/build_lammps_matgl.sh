#!/usr/bin/env bash
# Build LAMMPS with Kokkos (CUDA) and the ML-IAP Python coupling (mliappy),
# embedding the mlip environment's Python, for MatGL models.
#
# Usage:
#   KOKKOS_ARCH_FLAG=Kokkos_ARCH_AMPERE80 bash skills/mat-lammps-md/scripts/build_lammps_matgl.sh
#
# Settings (environment variables):
#   KOKKOS_ARCH_FLAG   Kokkos GPU architecture, from `nvidia-smi --query-gpu=compute_cap`
#                      (required; e.g. Kokkos_ARCH_AMPERE80, Kokkos_ARCH_HOPPER90)
#   LAMMPS_ROOT        sources and builds (default: ~/.cache/atomisticskills/lammps)
#   LAMMPS_REF         lammps/lammps tag (default: stable_22Jul2025_update4)
#   LAMMPS_BUILD_JOBS  parallel compile jobs (default: 16)
#   CLEAN_BUILD        1 to start from an empty build directory (default: 1)
#   CUDA_HOME          CUDA toolkit for a CUDA torch build (default: /usr/local/cuda)
#
# Requirements:
#   - git, cmake, g++, an MPI compiler wrapper (mpicxx), nvcc, an NVIDIA GPU
#   - the mlip environment running natively on this host (venv/run mlip+lammps)
#
# Result: $LAMMPS_ROOT/matgl/lmp. Run it in the same environment:
#   venv/run mlip+lammps $LAMMPS_ROOT/matgl/lmp -k on g 1 -sf kk -in in.file
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../../.." && pwd)"
RUN="${REPO}/venv/run"
LAMMPS_ROOT="${LAMMPS_ROOT:-${HOME}/.cache/atomisticskills/lammps}"
LAMMPS_SRC_DIR="${LAMMPS_ROOT}/src-lammps"
LAMMPS_BUILD_DIR="${LAMMPS_ROOT}/matgl"
LAMMPS_GIT_URL="${LAMMPS_GIT_URL:-https://github.com/lammps/lammps.git}"
LAMMPS_REF="${LAMMPS_REF:-stable_22Jul2025_update4}"
LAMMPS_BUILD_JOBS="${LAMMPS_BUILD_JOBS:-16}"
CLEAN_BUILD="${CLEAN_BUILD:-1}"
# PyTorch's CMake config looks for the CUDA toolkit (headers, cudart) of a CUDA
# torch build. Name it, rather than trust the first nvcc on PATH: some SDKs
# (NVIDIA HPC SDK) ship an nvcc whose directory has neither.
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
: "${KOKKOS_ARCH_FLAG:?set KOKKOS_ARCH_FLAG (e.g. Kokkos_ARCH_AMPERE80) from nvidia-smi --query-gpu=compute_cap}"

for cmd in git cmake g++ mpicxx nvidia-smi "${CUDA_HOME}/bin/nvcc"; do
  command -v "${cmd}" >/dev/null 2>&1 || { echo "Missing: ${cmd}" >&2; exit 1; }
done

# LAMMPS embeds the environment's Python, so it needs the environment on this
# host, not in a container.
ENV_PREFIX="$("${RUN}" mlip+lammps python -c 'import sys; print(sys.prefix)')"
if [[ "${ENV_PREFIX}" != "${REPO}/venv/mlip/.venv" ]]; then
  echo "The mlip environment does not run natively on this host (got ${ENV_PREFIX}); LAMMPS cannot be built against it here." >&2
  exit 1
fi
PYTHON_EXECUTABLE="${ENV_PREFIX}/bin/python"
CYTHONIZE_EXECUTABLE="${ENV_PREFIX}/bin/cythonize"
[[ -x "${CYTHONIZE_EXECUTABLE}" ]] || { echo "Missing ${CYTHONIZE_EXECUTABLE} (the lammps extra provides Cython)" >&2; exit 1; }

mkdir -p "${LAMMPS_ROOT}"
if [[ ! -d "${LAMMPS_SRC_DIR}/.git" ]]; then
  git clone --branch "${LAMMPS_REF}" --depth 1 "${LAMMPS_GIT_URL}" "${LAMMPS_SRC_DIR}"
fi
git -C "${LAMMPS_SRC_DIR}" fetch --depth 1 origin tag "${LAMMPS_REF}"
git -C "${LAMMPS_SRC_DIR}" checkout "${LAMMPS_REF}"

if [[ "${CLEAN_BUILD}" == "1" ]]; then
  rm -rf "${LAMMPS_BUILD_DIR}"
fi
mkdir -p "${LAMMPS_BUILD_DIR}"

cmake -S "${LAMMPS_SRC_DIR}/cmake" \
  -B "${LAMMPS_BUILD_DIR}" \
  -C "${LAMMPS_SRC_DIR}/cmake/presets/all_off.cmake" \
  -D CMAKE_BUILD_TYPE=Release \
  -D BUILD_MPI=ON \
  -D BUILD_OMP=ON \
  -D PKG_KOKKOS=ON \
  -D Kokkos_ENABLE_CUDA=ON \
  -D "${KOKKOS_ARCH_FLAG}=ON" \
  -D PKG_ML-IAP=ON \
  -D PKG_ML-SNAP=ON \
  -D PKG_PYTHON=ON \
  -D MLIAP_ENABLE_PYTHON=ON \
  -D CMAKE_CXX_STANDARD=17 \
  -D CMAKE_CXX_COMPILER=mpicxx \
  -D CUDA_TOOLKIT_ROOT_DIR="${CUDA_HOME}" \
  -D CUDAToolkit_ROOT="${CUDA_HOME}" \
  -D CMAKE_CUDA_COMPILER="${CUDA_HOME}/bin/nvcc" \
  -D Cythonize_EXECUTABLE="${CYTHONIZE_EXECUTABLE}" \
  -D Python_EXECUTABLE="${PYTHON_EXECUTABLE}"

cmake --build "${LAMMPS_BUILD_DIR}" -j "${LAMMPS_BUILD_JOBS}"

LMP_HELP_OUTPUT="$("${RUN}" mlip+lammps "${LAMMPS_BUILD_DIR}/lmp" -h 2>&1 || true)"
for pkg in KOKKOS ML-IAP PYTHON; do
  if [[ "${LMP_HELP_OUTPUT}" != *"${pkg}"* ]]; then
    echo "${pkg} not detected in lmp -h output." >&2
    exit 1
  fi
done

echo "Done: ${LAMMPS_BUILD_DIR}/lmp (run with: ${RUN} mlip+lammps ${LAMMPS_BUILD_DIR}/lmp ...)"
