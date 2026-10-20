#!/usr/bin/env bash
set -euo pipefail
scene_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
genesis_root="${GENESIS_ROOT:-/home/chloe/Desktop/genesis-local}"
cache_dir="${BEAR_CACHE_DIR:-$scene_dir/.cache}"
export GS_CACHE_FILE_PATH="$cache_dir/genesis"
export XDG_CACHE_HOME="$cache_dir"
export NUMBA_CACHE_DIR="$cache_dir/numba"
export MPLCONFIGDIR="$cache_dir/matplotlib"
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$cache_dir"
exec "$genesis_root/.venv/bin/python" "$scene_dir/scene.py" "$@"
