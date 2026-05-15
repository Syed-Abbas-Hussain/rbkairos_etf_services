#!/usr/bin/env bash
set -euo pipefail

PKG="rbkairos_etf_services"
MODEL_PATH="${MODEL_PATH:-$HOME/models/yolov5n_fruit.onnx}"
VISUALIZER="${VISUALIZER:-false}"
AUTO_GRASP="${AUTO_GRASP:-false}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

pids=()

cleanup() {
  echo
  echo "[run_session_with_vision] Stopping..."
  for pid in "${pids[@]:-}"; do
    kill -TERM "$pid" 2>/dev/null || true
  done
  sleep 1
  for pid in "${pids[@]:-}"; do
    kill -KILL "$pid" 2>/dev/null || true
  done
}
trap cleanup INT TERM EXIT

echo "[run_session_with_vision] Workspace: $WS_DIR"

# Source ROS + workspace
source /opt/ros/noetic/setup.bash
source "$WS_DIR/devel/setup.bash"

echo "[run_session_with_vision] Starting robot services..."
bash "$SCRIPT_DIR/run_services.sh" &
pids+=("$!")

sleep 3

echo "[run_session_with_vision] Starting vision pipeline..."
roslaunch "$PKG" vision_pipeline.launch \
  model_path:="$MODEL_PATH" \
  visualizer:="$VISUALIZER" \
  auto_grasp:="$AUTO_GRASP" &
pids+=("$!")

echo "[run_session_with_vision] Running. Ctrl+C to stop everything."
wait
