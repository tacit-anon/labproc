#!/usr/bin/env bash
# Full pipeline convenience wrapper. Equivalent to `tacit-annotate run` but exposes
# the underlying calls for shell-only environments.
#
# Usage:
#   ./scripts/run_pipeline.sh <video> <branch> [output_dir] [interval]
#
# Requires:
#   - ffmpeg / ffprobe in PATH
#   - tacit-annotator installed (`pip install -e .`)
#   - ANTHROPIC_API_KEY set in env

set -euo pipefail

VIDEO="${1:?usage: $0 <video> <branch> [output_dir] [interval]}"
BRANCH="${2:?usage: $0 <video> <branch> [output_dir] [interval]}"
OUTPUT_DIR="${3:-./out}"
INTERVAL="${4:-30}"

if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
  echo "error: ANTHROPIC_API_KEY not set" >&2
  exit 1
fi

if ! command -v tacit-annotate >/dev/null 2>&1; then
  echo "error: tacit-annotate not on PATH. Run: pip install -e ." >&2
  exit 1
fi

tacit-annotate run "$VIDEO" \
  --branch "$BRANCH" \
  --output-dir "$OUTPUT_DIR" \
  --interval "$INTERVAL"
