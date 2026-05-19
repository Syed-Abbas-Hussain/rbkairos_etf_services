#!/usr/bin/env bash
# Launch both robot planning stacks independently.
# Usage: ./bringup_sim_multi.sh [dry_run:=true] [domain_name:=fruit_collection_domain.rddl]
#
# Each robot runs in its own roslaunch process so their lifecycles are
# fully independent — one can be restarted without affecting the other.

set -e

ARGS="$@"

roslaunch rbkairos_etf_services robot1_sim.launch $ARGS &
PID1=$!

# Brief pause so robot1's prost_bridge claims port 2323 before robot2 starts.
sleep 5

roslaunch rbkairos_etf_services robot2_sim.launch $ARGS &
PID2=$!

echo "robot1 roslaunch PID: $PID1"
echo "robot2 roslaunch PID: $PID2"

# Wait for both; exit when either finishes (or Ctrl-C kills both).
trap "kill $PID1 $PID2 2>/dev/null; exit" INT TERM
wait $PID1 $PID2
