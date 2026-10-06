#!/usr/bin/env bash
# Build LAMMPS with the MACE pair style (ACEsuit's LAMMPS fork, ML-MACE)
# against the libtorch of the mlip environment.
#
# Usage:
#   bash skills/mat-lammps-md/scripts/build_lammps_mace.sh
#
# Settings (environment variables):
#   LAMMPS_ROOT        sources and builds (default: ~/.cache/atomisticskills/lammps)
#   LAMMPS_REF         ACEsuit/lammps branch or tag (default: mace)
#   LAMMPS_BUILD_JOBS  parallel compile jobs (default: 16)
#   CLEAN_BUILD        1 to start from an empty build directory (default: 1)
#   CUDA_HOME          CUDA toolkit for a CUDA torch build (default: /usr/local/cuda)
#
# Requirements:
#   - git, cmake, g++, an MPI compiler wrapper (mpicxx); on GPU hosts (a CUDA
#     torch build) a CUDA toolkit at least as new as torch's CUDA (12.6 for
#     the cu126 build, 13.0 for cu130)
#   - the mlip environment running natively on this host (venv/run mlip+lammps)
#
# Result: $LAMMPS_ROOT/mace/lmp. Run it in the same environment:
#   venv/run mlip+lammps $LAMMPS_ROOT/mace/lmp -in in.file
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../../.." && pwd)"
RUN="${REPO}/venv/run"
LAMMPS_ROOT="${LAMMPS_ROOT:-${HOME}/.cache/atomisticskills/lammps}"
LAMMPS_SRC_DIR="${LAMMPS_ROOT}/src-mace"
LAMMPS_BUILD_DIR="${LAMMPS_ROOT}/mace"
LAMMPS_GIT_URL="${LAMMPS_GIT_URL:-https://github.com/ACEsuit/lammps.git}"
LAMMPS_REF="${LAMMPS_REF:-mace}"
LAMMPS_BUILD_JOBS="${LAMMPS_BUILD_JOBS:-16}"
CLEAN_BUILD="${CLEAN_BUILD:-1}"
# PyTorch's CMake config looks for the CUDA toolkit (headers, cudart) of a CUDA
# torch build. Name it, rather than trust the first nvcc on PATH: some SDKs
# (NVIDIA HPC SDK) ship an nvcc whose directory has neither.
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

for cmd in git cmake g++ mpicxx; do
  command -v "${cmd}" >/dev/null 2>&1 || { echo "Missing: ${cmd}" >&2; exit 1; }
done

# The build links against the environment's own torch and MKL, so it needs the
# environment on this host, not in a container.
ENV_PREFIX="$("${RUN}" mlip+lammps python -c 'import sys; print(sys.prefix)')"
if [[ "${ENV_PREFIX}" != "${REPO}/venv/mlip/.venv" ]]; then
  echo "The mlip environment does not run natively on this host (got ${ENV_PREFIX}); LAMMPS cannot be built against it here." >&2
  exit 1
fi
PYTHON_EXECUTABLE="${ENV_PREFIX}/bin/python"
TORCH_CMAKE_PREFIX="$("${PYTHON_EXECUTABLE}" -c 'import torch; print(torch.utils.cmake_prefix_path)')"

# PyTorch's CMake config refuses a CUDA toolkit older than its own CUDA build
# ("Your installed Cuda version: 12.4 is too old, PyTorch requires CUDA 12.6").
TORCH_CUDA="$("${PYTHON_EXECUTABLE}" -c 'import torch; print(torch.version.cuda or "")')"
if [[ -n "${TORCH_CUDA}" ]]; then
  if [[ ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
    echo "torch in the mlip environment is a CUDA ${TORCH_CUDA} build, so this build needs a CUDA toolkit; none at CUDA_HOME=${CUDA_HOME}." >&2
    exit 1
  fi
  TOOLKIT_CUDA="$("${CUDA_HOME}/bin/nvcc" --version | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')"
  if [[ "$(printf '%s\n%s\n' "${TORCH_CUDA}" "${TOOLKIT_CUDA}" | sort -V | head -1)" != "${TORCH_CUDA}" ]]; then
    echo "torch in the mlip environment is a CUDA ${TORCH_CUDA} build, and PyTorch's CMake config needs a CUDA toolkit at least that new; CUDA_HOME=${CUDA_HOME} has ${TOOLKIT_CUDA}." >&2
    echo "Install a newer toolkit (NVIDIA's runfile installs into a user directory: --toolkit --toolkitpath=DIR) and set CUDA_HOME to it." >&2
    exit 1
  fi
fi

mkdir -p "${LAMMPS_ROOT}"
if [[ ! -d "${LAMMPS_SRC_DIR}/.git" ]]; then
  git clone --branch "${LAMMPS_REF}" --depth 1 "${LAMMPS_GIT_URL}" "${LAMMPS_SRC_DIR}"
fi
git -C "${LAMMPS_SRC_DIR}" fetch --depth 1 origin "${LAMMPS_REF}"
git -C "${LAMMPS_SRC_DIR}" checkout FETCH_HEAD

if [[ "${CLEAN_BUILD}" == "1" ]]; then
  rm -rf "${LAMMPS_BUILD_DIR}"
fi
mkdir -p "${LAMMPS_BUILD_DIR}"

# MKL comes from the lammps extra on x86_64 (PyTorch's CMake config asks for
# it there); aarch64 builds of torch do not use MKL. PyTorch >= 2.13 headers
# require C++20.
cmake -S "${LAMMPS_SRC_DIR}/cmake" \
  -B "${LAMMPS_BUILD_DIR}" \
  -D CMAKE_BUILD_TYPE=Release \
  -D BUILD_MPI=ON \
  -D BUILD_OMP=ON \
  -D PKG_OPENMP=ON \
  -D PKG_ML-MACE=ON \
  -D CMAKE_CXX_STANDARD=20 \
  -D CMAKE_CXX_COMPILER=mpicxx \
  -D CUDA_TOOLKIT_ROOT_DIR="${CUDA_HOME}" \
  -D CUDAToolkit_ROOT="${CUDA_HOME}" \
  -D Python_EXECUTABLE="${PYTHON_EXECUTABLE}" \
  -D CMAKE_PREFIX_PATH="${TORCH_CMAKE_PREFIX};${ENV_PREFIX}" \
  -D MKL_ROOT="${ENV_PREFIX}"

cmake --build "${LAMMPS_BUILD_DIR}" -j "${LAMMPS_BUILD_JOBS}"

LMP_HELP_OUTPUT="$("${RUN}" mlip+lammps "${LAMMPS_BUILD_DIR}/lmp" -h 2>&1 || true)"
if [[ "${LMP_HELP_OUTPUT}" != *"ML-MACE"* ]]; then
  echo "ML-MACE not detected in lmp -h output." >&2
  exit 1
fi

echo "Done: ${LAMMPS_BUILD_DIR}/lmp (run with: ${RUN} mlip+lammps ${LAMMPS_BUILD_DIR}/lmp ...)"
