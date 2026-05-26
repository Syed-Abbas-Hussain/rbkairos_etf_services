#!/usr/bin/env python3

import json
import os

import numpy as np
import rospkg
import rospy

from prost_ros.msg import KeyValue
from prost_ros.srv import StartPlanning, SubmitObservation
from rbkairos_etf_services.srv import ActionServer

# Package Imports
from rbkairos_etf_services.evaluator import FruitHarvestingRewardEvaluator


def get_problem_data_paths(
    package_name="rbkairos_etf_services",
    domain_name="fruit_collection_domain.rddl",
    instance_name="instance_new_real.rddl",
    actions_name="actions.json",
):
    rp = rospkg.RosPack()
    pkg_path = rp.get_path(package_name)
    problem_data = os.path.join(pkg_path, "problem_data")
    return (
        os.path.join(problem_data, domain_name),
        os.path.join(problem_data, instance_name),
        os.path.join(problem_data, actions_name),
    )


class ServiceCaller:
    TERMINAL_PLANNER_ACTIONS = {"ERROR"}
    RESTART_PLANNER_ACTIONS  = {"ROUND_END", "FAILED"}
    MAX_PLANNING_RESTARTS    = 5

    def __init__(self, domain_file, instance_file, actions_file):
        bridge_ns = rospy.get_param("~prost_bridge_ns", "/prost_bridge")

        rospy.wait_for_service(f"{bridge_ns}/start_planning")
        rospy.wait_for_service(f"{bridge_ns}/submit_observation")

        self.start_planning = rospy.ServiceProxy(f"{bridge_ns}/start_planning", StartPlanning)
        self.submit_obs = rospy.ServiceProxy(f"{bridge_ns}/submit_observation", SubmitObservation)

        with open(domain_file, 'r') as f:
            self.domain = f.read()
        with open(instance_file, 'r') as f:
            self.instance = f.read()

        rospy.loginfo("Sending the Planning request")
        self.resp = self.start_planning(self.domain, self.instance, 60)
        if not self.resp.success:
            rospy.logerr("PROST server failed")

        self.evaluator = FruitHarvestingRewardEvaluator(domain_file, instance_file)

        self.obs = self.evaluator.get_initial_state()
        self.reward = 0.0

        # Load the actions.json action-to-robot motion data
        with open(actions_file, "r", encoding="utf-8") as f:
            self.ACTION_DATA = json.load(f)

        # Object index maps (ordered by RDDL instance definition)
        self.positions = self.evaluator.positions
        self.locations = self.evaluator.locations
        self.robot_names = self.evaluator.robots
        self.robot_to_idx = self.evaluator.robot_to_idx

        self.pos_to_idx = {name: i for i, name in enumerate(self.positions)}
        self.loc_to_idx = {name: i for i, name in enumerate(self.locations)}

        self.idle_count = 0
        self._planning_restarts = 0

        # When dry_run=True the node skips all action-server calls and treats
        # every action as instantly successful.  Useful for testing the planner
        # loop without a running action_manager.
        self.dry_run = rospy.get_param("~dry_run", False)
        if self.dry_run:
            rospy.logwarn("DRY RUN mode: action server calls are SKIPPED, all actions assumed successful.")

        self.manual_input = rospy.get_param("~manual_input", False)
        if self.manual_input:
            rospy.logwarn("MANUAL INPUT mode: you will be prompted after each action (s=success, f=failure).")

        # Per-robot action server proxies — created lazily on first use
        self._robot_action_proxies = {}

    def _get_robot_proxy(self, robot_name: str):
        if robot_name not in self._robot_action_proxies:
            service_name = f"/{robot_name}/action_server"
            rospy.loginfo(f"Waiting for action server: {service_name}")
            rospy.wait_for_service(service_name)
            self._robot_action_proxies[robot_name] = rospy.ServiceProxy(service_name, ActionServer)
        return self._robot_action_proxies[robot_name]

    # ------------------------------------------------------------------
    # Action parsing
    # ------------------------------------------------------------------

    def parse_action(self, action_name, action_params):
        """Return (robot_idx, aisle_index, location_index) for one robot action."""
        robot_idx = 0
        aisle_index = -1
        location_index = -1

        if action_params:
            maybe_robot = action_params[0]
            robot_idx = self.robot_to_idx.get(maybe_robot, robot_idx)

        if action_name == "navigate" and len(action_params) >= 2:
            aisle_index = self.pos_to_idx.get(action_params[1], -1)
        elif action_name in ("grasp_fruit", "load_to_bin") and len(action_params) >= 2:
            location_index = self.loc_to_idx.get(action_params[1], -1)

        return robot_idx, aisle_index, location_index

    # ------------------------------------------------------------------
    # Observation submission helpers
    # ------------------------------------------------------------------

    def append_scalar_fluent(self, obs_list, name, value):
        obs_list.append(KeyValue(name, "true" if bool(value) else "false"))

    def append_vector_fluent(self, obs_list, name, values, objects):
        for idx, val in enumerate(values):
            obs_list.append(KeyValue(
                f"{name}({objects[idx]})",
                "true" if bool(val) else "false",
            ))

    def append_robot_fluent(self, obs_list, name, values, objects):
        for robot_idx, robot_name in enumerate(self.robot_names):
            for obj_idx, obj_name in enumerate(objects):
                obs_list.append(KeyValue(
                    f"{name}({robot_name},{obj_name})",
                    "true" if bool(values[robot_idx, obj_idx]) else "false",
                ))

    def append_robot_scalar_fluent(self, obs_list, name, values):
        for robot_idx, robot_name in enumerate(self.robot_names):
            obs_list.append(KeyValue(
                f"{name}({robot_name})",
                "true" if bool(values[robot_idx]) else "false",
            ))

    # ------------------------------------------------------------------
    # Single-robot action dispatch
    # ------------------------------------------------------------------

    def _dispatch_robot_action(self, action_name, action_params):
        """
        Resolve motion data and dispatch one robot action to the appropriate
        physical action server.  Returns (robot_idx, action_name, action_index).

        All actions are assumed to succeed (caller guarantee).
        """
        robot_idx, aisle_index, location_index = self.parse_action(action_name, action_params)
        robot_name = self.robot_names[robot_idx]

        real_action = np.zeros(6, dtype=float)
        action_index = -1

        if action_name == "navigate":
            if 0 <= aisle_index < len(self.ACTION_DATA.get(action_name, [])):
                real_action = self.ACTION_DATA[action_name][aisle_index]
            else:
                rospy.logwarn(f"No motion entry for navigate target index {aisle_index}.")
            action_index = aisle_index

        elif action_name == "grasp_fruit":
            if 0 <= location_index < len(self.ACTION_DATA.get(action_name, [])):
                real_action = self.ACTION_DATA[action_name][location_index]
            else:
                rospy.logwarn(f"No motion entry for grasp target index {location_index}.")
            action_index = location_index

        elif action_name == "load_to_bin":
            real_action = self.ACTION_DATA[action_name][0]
            action_index = location_index

        elif action_name == "unload":
            # The robot is already at the unload station when this action fires.
            # Reuse the navigate pose for that position so the action_manager
            # can do the final dock-and-tip sequence, consistent with how it
            # handles the unload action (action_navigate).
            current_pos = np.where(self.obs["robot_at"][robot_idx])[0]
            if current_pos.size > 0:
                pos_idx = int(current_pos[0])
                nav_data = self.ACTION_DATA.get("navigate", [])
                if 0 <= pos_idx < len(nav_data):
                    real_action = nav_data[pos_idx]
                else:
                    rospy.logwarn(f"No navigate entry for unload station at position index {pos_idx}.")

        elif action_name in ("wait", "NOOP"):
            pass

        else:
            rospy.logwarn(f"Unknown planner action '{action_name}'.")

        # Execute on the physical robot
        if action_name not in ("wait", "NOOP") and not self.dry_run:
            try:
                proxy = self._get_robot_proxy(robot_name)
                proxy(action_name, real_action, 50.0)
            except Exception as e:
                rospy.logwarn(f"Action server call failed for {robot_name}: {e}")

        return robot_idx, action_name, action_index

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        while True:
            for robot_idx, robot_name in enumerate(self.robot_names):
                robot_row = list(self.obs["robot_at"][robot_idx])
                true_count = sum(1 for v in robot_row if v)
                if true_count != 1:
                    rospy.logerr(f"{robot_name} robot_at has {true_count} true values — expected 1.")

            rospy.loginfo("Sending observations and reward to planner...")

            obs_to_submit = []
            self.append_scalar_fluent(obs_to_submit, "all_fruits_done", self.obs["all_fruits_done"])
            self.append_robot_fluent(obs_to_submit, "robot_at", self.obs["robot_at"], self.positions)
            self.append_vector_fluent(obs_to_submit, "fruit_at", self.obs["fruit_at"], self.locations)
            self.append_robot_fluent(
                obs_to_submit, "fruit_collected", self.obs["fruit_collected"], self.locations
            )
            self.append_robot_fluent(
                obs_to_submit, "fruit_in_bin", self.obs["fruit_in_bin"], self.locations
            )
            self.append_vector_fluent(
                obs_to_submit, "fruits_unloaded", self.obs["fruits_unloaded"], self.locations
            )
            self.append_vector_fluent(
                obs_to_submit, "position_visited", self.obs["position_visited"], self.positions
            )
            if "prev_position" in self.obs:
                self.append_robot_fluent(
                    obs_to_submit, "prev_position", self.obs["prev_position"], self.positions
                )

            response = self.submit_obs(obs_to_submit, self.reward)
            action_name = response.action_name
            action_data = response.action_params

            rospy.loginfo(f"Planner response: action_name={action_name}, params={list(action_data)}")

            if action_name in self.TERMINAL_PLANNER_ACTIONS:
                rospy.logerr(
                    f"Planner returned terminal status '{action_name}' with params {list(action_data)}. "
                    "Stopping execution loop."
                )
                break

            # Decode joint (multi-robot) or single-robot action
            if action_name == "JOINT":
                # Bridge encoded all robot actions as JSON in action_data[0]
                robot_action_list = json.loads(action_data[0])
                robot_actions = [(entry[0], entry[1:]) for entry in robot_action_list]
            else:
                robot_actions = [(action_name, list(action_data))]

            rospy.loginfo(f"Dispatching {len(robot_actions)} robot action(s)...")

            observed_action = self.evaluator.create_action_template()

            for single_action_name, single_action_params in robot_actions:
                rospy.loginfo(f"  {single_action_name} {single_action_params}")
                robot_idx, _, action_index = self._dispatch_robot_action(
                    single_action_name, single_action_params
                )

                if single_action_name == "NOOP":
                    continue

                if self.manual_input:
                    key = input(f"  [{single_action_name} {single_action_params}] success? [s/f]: ").strip().lower()
                    success = (key == "s")
                    rospy.loginfo(f"  -> {'SUCCESS' if success else 'FAILURE'}")
                else:
                    success = True

                if single_action_name == "unload":
                    observed_action["unload"][robot_idx] = success
                elif single_action_name == "wait":
                    observed_action["wait"][robot_idx] = success
                elif action_index >= 0:
                    observed_action[single_action_name][robot_idx][action_index] = success
                else:
                    rospy.logwarn(
                        f"Skipping state update for '{single_action_name}': target index not resolved."
                    )

            next_obs = self.evaluator.step(self.obs, observed_action)
            reward = self.evaluator.evaluate_reward(self.obs, observed_action, next_obs)

            if self.idle_count > 200 or next_obs["all_fruits_done"]:
                reward = 0.0
                rospy.loginfo("Sim finished")
                break

            self.obs = next_obs
            self.reward = reward
            self.idle_count += 1


if __name__ == "__main__":
    rospy.init_node("prost_service_caller")

    startup_delay = rospy.get_param("~startup_delay", 0.0)
    if startup_delay > 0:
        rospy.loginfo(f"Startup delay: {startup_delay}s — waiting for other planners to initialize.")
        rospy.sleep(startup_delay)

    domain_file, instance_file, actions_file = get_problem_data_paths(
        domain_name=rospy.get_param("~domain_name", "fruit_collection_domain.rddl"),
        instance_name=rospy.get_param("~instance_name", "instance_new_real.rddl"),
    )

    rospy.loginfo(f"Domain:   {domain_file}")
    rospy.loginfo(f"Instance: {instance_file}")

    sc = ServiceCaller(domain_file, instance_file, actions_file)
    sc.run()
