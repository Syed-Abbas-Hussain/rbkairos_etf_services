#!/usr/bin/env python3

import json
import os

import numpy as np
import rospkg
import rospy

from prost_ros.msg import KeyValue
from prost_ros.srv import StartPlanning, SubmitObservation
from rbkairos_etf_services.srv import ActionServer

# Usage: rosrun prost_ros prost_bridge.py _prost_path:=/home/ruzamladji/catkin_ws/src/prost_ros/prost/prost.py

# Package Imports
from rbkairos_etf_services.evaluator import FruitHarvestingRewardEvaluator

def get_problem_data_paths(
    package_name="rbkairos_etf_services",
    domain_name="domain_new.rddl",
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
    TERMINAL_PLANNER_ACTIONS = {"FAILED", "ROUND_END", "ERROR"}

    MODE_REAL = "real"
    MODE_SIMULATION = "simulation"
    MODE_GAZEBO = "gazebo"

    def __init__(self, robot_id, domain_file, instance_file, actions_file):
        rospy.init_node("prost_service_caller")

        self.mode = rospy.get_param("~mode", self.MODE_REAL)
        robot_namespaces_param = rospy.get_param("~robot_namespaces", "")
        self.robot_namespaces = [ns.strip() for ns in robot_namespaces_param.split(",") if ns.strip()]

        rospy.wait_for_service('/prost_bridge/start_planning')
        rospy.wait_for_service('/prost_bridge/submit_observation')

        self.start_planning = rospy.ServiceProxy('/prost_bridge/start_planning', StartPlanning)
        self.submit_obs = rospy.ServiceProxy('/prost_bridge/submit_observation', SubmitObservation)

        if self.mode == self.MODE_REAL:
            rospy.wait_for_service('/action_server')
            self.perform_action = rospy.ServiceProxy('/action_server', ActionServer)
            self.robot_action_servers = []
            rospy.loginfo("Running in REAL mode — single robot hardware control.")
        elif self.mode == self.MODE_SIMULATION:
            self.perform_action = None
            self.robot_action_servers = []
            rospy.loginfo("Running in SIMULATION mode — hardware calls skipped, evaluator drives state.")
        elif self.mode == self.MODE_GAZEBO:
            self.perform_action = None
            self.robot_action_servers = []
            for ns in self.robot_namespaces:
                srv_name = f"/{ns}/action_server"
                rospy.wait_for_service(srv_name)
                self.robot_action_servers.append(rospy.ServiceProxy(srv_name, ActionServer))
            rospy.loginfo(f"Running in GAZEBO mode — {len(self.robot_action_servers)} robot(s) connected.")
        else:
            rospy.logwarn(f"Unknown mode '{self.mode}', falling back to 'real'.")
            self.mode = self.MODE_REAL
            rospy.wait_for_service('/action_server')
            self.perform_action = rospy.ServiceProxy('/action_server', ActionServer)
            self.robot_action_servers = []

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

        # Load the actions.json action to experiment data
        with open(actions_file, "r", encoding="utf-8") as f:
            self.ACTION_DATA = json.load(f)

        # Create mappings from indices to object names (and vice-versa) based on the environment model
        # This assumes pyRDDLGym state vectors are ordered by the object indices.
        self.positions = self.evaluator.positions
        self.locations = self.evaluator.locations
        self.robot_names = self.evaluator.robots
        self.robot_to_idx = self.evaluator.robot_to_idx
        self.idle_count = 0

        self.pos_to_idx = {name: i for i, name in enumerate(self.positions)}
        self.loc_to_idx = {name: i for i, name in enumerate(self.locations)}
        self.default_robot_idx = self.robot_to_idx.get(robot_id, 0)

    def _execute_hardware_action(self, robot_idx, action_name, real_action):
        if self.mode == self.MODE_SIMULATION:
            return True
        elif self.mode == self.MODE_REAL:
            response = self.perform_action(action_name, real_action, 50.0)
            return response.success
        elif self.mode == self.MODE_GAZEBO:
            if 0 <= robot_idx < len(self.robot_action_servers):
                response = self.robot_action_servers[robot_idx](action_name, real_action, 50.0)
                return response.success
            rospy.logwarn(f"No action server for robot_idx={robot_idx} (have {len(self.robot_action_servers)}).")
            return False
        return False

    def parse_action(self, action_name, action_params):
        robot_idx = self.default_robot_idx
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

    def append_scalar_fluent(self, obs_to_submit, name, value):
        obs_to_submit.append(KeyValue(name, "true" if bool(value) else "false"))

    def append_vector_fluent(self, obs_to_submit, name, values, objects):
        for idx, val in enumerate(values):
            obs_to_submit.append(
                KeyValue(
                    f"{name}({objects[idx]})",
                    "true" if bool(val) else "false",
                )
            )

    def append_robot_fluent(self, obs_to_submit, name, values, objects):
        for robot_idx, robot_name in enumerate(self.robot_names):
            for obj_idx, obj_name in enumerate(objects):
                obs_to_submit.append(
                    KeyValue(
                        f"{name}({robot_name},{obj_name})",
                        "true" if bool(values[robot_idx, obj_idx]) else "false",
                    )
                )

    def append_robot_scalar_fluent(self, obs_to_submit, name, values):
        for robot_idx, robot_name in enumerate(self.robot_names):
            obs_to_submit.append(
                KeyValue(
                    f"{name}({robot_name})",
                    "true" if bool(values[robot_idx]) else "false",
                )
            )

    def run(self):
        while True:
            #rospy.logwarn("=== STATE SNAPSHOT ===")
            #rospy.logwarn(f"fruit_at:         {list(zip(self.locations, self.obs['fruit_at']))}")
            #rospy.logwarn(f"position_visited: {list(zip(self.positions, self.obs['position_visited']))}")
            #rospy.logwarn(f"fruits_unloaded:  {list(zip(self.locations, self.obs['fruits_unloaded']))}")
            #rospy.logwarn(f"all_fruits_done:  {self.obs['all_fruits_done']}")

            #for robot_idx, robot_name in enumerate(self.robot_names):
                #rospy.logwarn(
                #    f"{robot_name} robot_at:        "
                #    f"{list(zip(self.positions, self.obs['robot_at'][robot_idx]))}"
                #)
                #rospy.logwarn(
                #    f"{robot_name} fruit_collected: "
                #    f"{list(zip(self.locations, self.obs['fruit_collected'][robot_idx]))}"
                #)
                #rospy.logwarn(
                #    f"{robot_name} fruit_in_bin:    "
                #    f"{list(zip(self.locations, self.obs['fruit_in_bin'][robot_idx]))}"
                #)

            for robot_idx, robot_name in enumerate(self.robot_names):
                robot_row = list(self.obs["robot_at"][robot_idx])
                robot_pos_idx = robot_row.index(True) if True in robot_row else -1
                #rospy.logwarn(
                #    f"{robot_name} is at: "
                #    f"{self.positions[robot_pos_idx] if robot_pos_idx >= 0 else 'NOWHERE - BUG!'}"
                #)
                true_count = sum(1 for value in robot_row if value)
                if true_count != 1:
                    rospy.logerr(f"{robot_name} robot_at has {true_count} true values - expected 1.")
            
            rospy.loginfo("Sending Observations and rewards to the planner . . .")

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

            #for kv in obs_to_submit[:20]:
            #    rospy.loginfo(f"SEND: {kv.key} = {kv.value}")
            #rospy.loginfo(f"Total KeyValues: {len(obs_to_submit)}, reward={float(self.reward)}")
            
            action_to_take = self.submit_obs(obs_to_submit, self.reward)
            action_name = action_to_take.action_name
            action_data = action_to_take.action_params

            rospy.loginfo(f"Planner action: {action_name} {list(action_data)}")

            if action_name in self.TERMINAL_PLANNER_ACTIONS:
                rospy.logerr(
                    f"Planner returned terminal status '{action_name}' with params {list(action_data)}. "
                    "Stopping execution loop."
                )
                break

            robot_idx, aisle_index, location_index = self.parse_action(action_name, action_data)

            rospy.loginfo("Performing action . . .")

            action_index = -1
            real_action = np.array([0, 0, 0, 0, 0, 0], dtype=float)

            if action_name == "navigate":
                if 0 <= aisle_index < len(self.ACTION_DATA[action_name]):
                    real_action = self.ACTION_DATA[action_name][aisle_index]
                else:
                    rospy.logwarn(f"No motion entry for navigate target index {aisle_index}.")
                action_index = aisle_index

            elif action_name == "grasp_fruit":
                if 0 <= location_index < len(self.ACTION_DATA[action_name]):
                    real_action = self.ACTION_DATA[action_name][location_index]
                else:
                    rospy.logwarn(f"No motion entry for grasp target index {location_index}.")
                action_index = location_index

            elif action_name == "load_to_bin":
                real_action = self.ACTION_DATA[action_name][0]
                action_index = location_index

            elif action_name == "unload":
                unload_station_indices = np.where(self.evaluator.unload_station)[0]
                current_station_idx = -1
                current_pos = np.where(self.obs["robot_at"][robot_idx])[0]
                if current_pos.size > 0 and self.evaluator.unload_station[current_pos[0]]:
                    matching = np.where(unload_station_indices == current_pos[0])[0]
                    if matching.size > 0:
                        current_station_idx = int(matching[0])
                if 0 <= current_station_idx < len(self.ACTION_DATA[action_name]):
                    real_action = self.ACTION_DATA[action_name][current_station_idx]
                else:
                    rospy.logwarn("Could not map unload action to an unload-station motion entry.")
            elif action_name == "wait":
                pass

            elif action_name == "NOOP":
                pass

            else:
                rospy.logwarn(f"Unknown planner action '{action_name}'.")

            success = self._execute_hardware_action(robot_idx, action_name, real_action)
            # rospy.loginfo(f"Action finished with success: {success} | {response.message}")

            observed_action = self.evaluator.create_action_template()

            if action_name != "NOOP":
                if action_name == "unload":
                    observed_action[action_name][robot_idx] = success
                elif action_name == "wait":
                    observed_action[action_name][robot_idx] = success
                else:
                    if action_index >= 0:
                        observed_action[action_name][robot_idx][action_index] = success
                    else:
                        rospy.logwarn(
                            f"Skipping action state update for '{action_name}' because the target index was not resolved."
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

    domain_file, instance_file, actions_file = get_problem_data_paths()


    print(domain_file)
    print(instance_file)

    # domain_file = "/problem_data/domain.rddl"
    # instance_file = "/problem_data/instance.rddl"
    
    sc = ServiceCaller("Robotnik", domain_file, instance_file, actions_file)
    sc.run()
