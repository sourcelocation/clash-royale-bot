#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/cpp/build"
PYBIND_CMAKE_DIR="$(python -m pybind11 --cmakedir 2>/dev/null || true)"

mkdir -p "${BUILD_DIR}"

CMAKE_ARGS=(
  -S "${ROOT_DIR}/cpp"
  -B "${BUILD_DIR}"
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_LIBRARY_OUTPUT_DIRECTORY="${ROOT_DIR}"
)

if [[ -n "${PYBIND_CMAKE_DIR}" ]]; then
  CMAKE_ARGS+=( -Dpybind11_DIR="${PYBIND_CMAKE_DIR}" )
else
  echo "pybind11 Python package not found; relying on CMake package discovery"
fi

cmake "${CMAKE_ARGS[@]}"
cmake --build "${BUILD_DIR}" -j

echo "Built knockoff_cr_cpp module at: ${ROOT_DIR}/knockoff_cr_cpp$(python - <<'PY'
import sysconfig
print(sysconfig.get_config_var('EXT_SUFFIX') or '.so')
PY
)"
