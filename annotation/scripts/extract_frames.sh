#!/bin/bash
# Extract frames from a video at exact timestamp intervals using -ss.
# DO NOT replace -ss with the fps= filter — that drifts and silently corrupts timestamps.
#
# Usage:
#   extract_frames.sh <video> <output_dir> [interval_seconds=30]
#
# Output:
#   <output_dir>/t{seconds_padded5}.jpg — one file per sampled timestamp.

set -e

VIDEO="$1"
OUT_DIR="$2"
INTERVAL="${3:-30}"

if [ -z "$VIDEO" ] || [ -z "$OUT_DIR" ]; then
  echo "Usage: $0 <video> <output_dir> [interval_seconds=30]" >&2
  exit 1
fi

if [ ! -f "$VIDEO" ]; then
  echo "Error: video not found: $VIDEO" >&2
  exit 2
fi

mkdir -p "$OUT_DIR"

# Get integer duration via ffprobe
DURATION=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$VIDEO" 2>/dev/null | awk '{print int($1)}')

if [ -z "$DURATION" ] || [ "$DURATION" -le 0 ]; then
  echo "Error: could not determine duration for $VIDEO" >&2
  exit 3
fi

echo "Video:    $VIDEO"
echo "Duration: ${DURATION}s"
echo "Interval: ${INTERVAL}s"
echo "Output:   $OUT_DIR"
echo ""

TS=0
COUNT=0
while [ "$TS" -le "$DURATION" ]; do
  OUT=$(printf "%s/t%05d.jpg" "$OUT_DIR" "$TS")
  ffmpeg -y -ss "$TS" -i "$VIDEO" -frames:v 1 -q:v 2 "$OUT" -loglevel error
  printf "  t=%5ds  ->  %s\n" "$TS" "$(basename "$OUT")"
  TS=$((TS + INTERVAL))
  COUNT=$((COUNT + 1))
done

echo ""
echo "Extracted $COUNT frames at native resolution."
