#!/usr/bin/env bash
# Renders the metasphere images in docs/assets/img with the viewer, and saves the two hierarchies
# the builder comparison measures (docs/_figures/mlod_stats.py).
#
#   bash docs/_figures/render_images.sh [path/to/meshlets]
#
# MLOD_DIR=dir keeps the two .mlod files there (for mlod_stats.py); otherwise they are temporary.
#
# Needs ffmpeg (PNG -> JPEG / WebP). Run from the repository root.
set -euo pipefail

VIEWER=${1:-./build/meshlets}
IMG=docs/assets/img
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
MLOD=${MLOD_DIR:-$TMP}
mkdir -p "$MLOD"

# both hierarchies at default settings; the renders below load them, so every image of one builder
# shows the same DAG
"$VIEWER" --no-ui --frames 1 --save-lod "$MLOD/meshoptimizer.mlod"
"$VIEWER" --no-ui --frames 1 --lod-source meshlet-optimizer --save-lod "$MLOD/meshlet-optimizer.mlod"

# render NAME VIEWER-ARGS...: 900x900 on white, stored as a 540x540 JPEG
render() {
    local name=$1
    shift
    "$VIEWER" --no-ui --width 900 --height 900 --background 1 1 1 --screenshot "$TMP/$name.png" "$@"
    ffmpeg -loglevel error -y -i "$TMP/$name.png" -vf scale=540:540:flags=lanczos -q:v 3 "$IMG/$name.jpg"
}

# DAG levels of both builders
for level in 0 2 4 6; do
    render "lod$level" "$MLOD/meshoptimizer.mlod" --lod-level "$level"
    render "mopt-lod$level" "$MLOD/meshlet-optimizer.mlod" --lod-level "$level"
done

# automatic selection from a closer view, same camera and threshold for both
AUTO=(--lod-error 4 --yaw 70 --pitch 25 --zoom 1.15)
render lod-auto "$MLOD/meshoptimizer.mlod" "${AUTO[@]}"
render mopt-lod-auto "$MLOD/meshlet-optimizer.mlod" "${AUTO[@]}"

# meshlet-optimizer's per-level UV atlases (metasphere has no UVs of its own)
render mopt-uv0 "$MLOD/meshlet-optimizer.mlod" --lod-level 0 --uv-grid --uv-grid-cells 48
render mopt-uv4 "$MLOD/meshlet-optimizer.mlod" --lod-level 4 --uv-grid --uv-grid-cells 48

# single-level builders
for algorithm in scan greedy spatial; do
    render "algo-$algorithm" --no-lod -a "$algorithm"
done

# the viewer with its sidebar, for the README
"$VIEWER" --width 1600 --height 1000 --lod-source meshlet-optimizer --screenshot "$TMP/screenshot.png"
ffmpeg -loglevel error -y -i "$TMP/screenshot.png" -quality 85 "$IMG/screenshot.webp"

ls -l "$IMG"
