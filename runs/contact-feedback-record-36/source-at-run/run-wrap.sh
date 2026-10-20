#!/usr/bin/env bash
set -euo pipefail
scene_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$scene_dir/run.sh" --task wrap --steps 50 --stop-after thread_arms "$@"
