#!/usr/bin/env bash
set -euo pipefail

# ---- Config ----
NS="robot"
PKG="rbkairos_etf_services"
LOG_DIR="${LOG_DIR:-./logs}"
mkdir -p "$LOG_DIR"

pids=()

cleanup() {
  echo
  echo "[run_services] Stopping services..."
  for pid in "${pids[@]:-}"; do
    kill -TERM "$pid" 2>/dev/null || true
  done
  sleep 1
  for pid in "${pids[@]:-}"; do
    kill -KILL "$pid" 2>/dev/null || true
  done
  echo "[run_services] Done."
}
trap cleanup INT TERM EXIT

run_node() {
  local name="$1"; shift
  echo "[run_services] Starting: $name (ROS_NAMESPACE=$NS)"
  env ROS_NAMESPACE="$NS" rosrun "$PKG" "$@" > "${LOG_DIR}/${name}.log" 2>&1 &
  pids+=("$!")
  echo "[run_services]  -> PID ${pids[-1]} (logs: ${LOG_DIR}/${name}.log)"
}

run_node_no_ns() {
  local name="$1"; shift
  echo "[run_services] Starting: $name (NO ROS_NAMESPACE)"
  rosrun "$PKG" "$@" > "${LOG_DIR}/${name}.log" 2>&1 &
  pids+=("$!")
  echo "[run_services]  -> PID ${pids[-1]} (logs: ${LOG_DIR}/${name}.log)"
}

# ---- Start nodes ----
run_node       "move_base_service_node" move_rbkairos.py
run_node       "move_arm_service_node"  move_moveit_fr3.py
run_node_no_ns "action_server_node"     action_manager.py

echo "[run_services] Started. Tail logs with: tail -f ${LOG_DIR}/*.log"
echo "[run_services] Ctrl+C to stop."
wait
