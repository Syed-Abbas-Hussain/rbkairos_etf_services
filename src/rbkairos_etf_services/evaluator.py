from typing import Any, Dict, Tuple

import numpy as np
from pyRDDLGym.core.env import RDDLEnv


class FruitHarvestingRewardEvaluator:
    """Evaluator and lightweight simulator for the robot-indexed fruit domain."""

    def __init__(self, domain_file: str, instance_file: str):
        self.env = RDDLEnv(domain=domain_file, instance=instance_file)
        env_model = self.env.model

        self.robots = self._get_sorted_objects("robot")
        self.positions = self._get_sorted_objects("aisle_position")
        self.locations = self._get_sorted_objects("location")

        self.num_robots = len(self.robots)
        self.num_positions = len(self.positions)
        self.num_locations = len(self.locations)

        self.robot_to_idx = {name: idx for idx, name in enumerate(self.robots)}
        self.position_to_idx = {name: idx for idx, name in enumerate(self.positions)}
        self.location_to_idx = {name: idx for idx, name in enumerate(self.locations)}

        self.max_capacity = float(env_model._non_fluents["MAX_CAPACITY"])
        self.capacity_threshold = float(env_model._non_fluents["CAPACITY_THRESHOLD"])
        self.fruit_weights = np.asarray(env_model._non_fluents["fruit_weight"], dtype=float)
        self.fruit_ripe = np.asarray(env_model._non_fluents["fruit_ripe"], dtype=bool)

        self.adjacent = self._reshape_bool(
            env_model._non_fluents["adjacent"], (self.num_positions, self.num_positions)
        )
        self.distance = self._reshape_float(
            env_model._non_fluents["distance"], (self.num_positions, self.num_positions)
        )
        self.reachable_from = self._reshape_bool(
            env_model._non_fluents["reachable_from"], (self.num_locations, self.num_positions)
        )
        self.unload_station = np.asarray(env_model._non_fluents["unload_station"], dtype=bool)
        self.discount = env_model._discount

        # Whether the domain tracks previous position (new domain feature)
        self._has_prev_position = "prev_position" in env_model._state_fluents

    def _get_sorted_objects(self, type_name: str):
        env_model = self.env.model
        objs = [obj for obj, obj_type in env_model._object_to_type.items() if obj_type == type_name]
        return sorted(objs, key=lambda obj: env_model._object_to_index[obj])

    def _reshape_bool(self, values: Any, shape: Tuple[int, ...]) -> np.ndarray:
        return np.asarray(values, dtype=bool).reshape(shape)

    def _reshape_float(self, values: Any, shape: Tuple[int, ...]) -> np.ndarray:
        return np.asarray(values, dtype=float).reshape(shape)

    def create_state_template(self) -> Dict[str, Any]:
        state = {
            "all_fruits_done": False,
            "fruit_at": np.zeros(self.num_locations, dtype=bool),
            "fruit_collected": np.zeros((self.num_robots, self.num_locations), dtype=bool),
            "fruit_in_bin": np.zeros((self.num_robots, self.num_locations), dtype=bool),
            "fruits_unloaded": np.zeros(self.num_locations, dtype=bool),
            "position_visited": np.zeros(self.num_positions, dtype=bool),
            "robot_at": np.zeros((self.num_robots, self.num_positions), dtype=bool),
        }
        if self._has_prev_position:
            state["prev_position"] = np.zeros((self.num_robots, self.num_positions), dtype=bool)
        return state

    def create_action_template(self) -> Dict[str, Any]:
        return {
            "grasp_fruit": np.zeros((self.num_robots, self.num_locations), dtype=bool),
            "load_to_bin": np.zeros((self.num_robots, self.num_locations), dtype=bool),
            "navigate": np.zeros((self.num_robots, self.num_positions), dtype=bool),
            "unload": np.zeros(self.num_robots, dtype=bool),
            "wait": np.zeros(self.num_robots, dtype=bool),
        }

    def create_observation_template(self) -> Dict[str, Any]:
        obs = self.create_state_template()
        obs.update(self.create_action_template())
        return obs

    def get_initial_state(self) -> Dict[str, Any]:
        state = self.create_state_template()
        raw = self.env.model._state_fluents
        state["all_fruits_done"] = bool(raw["all_fruits_done"])
        state["fruit_at"] = np.asarray(raw["fruit_at"], dtype=bool)
        state["fruit_collected"] = self._reshape_bool(
            raw["fruit_collected"], (self.num_robots, self.num_locations)
        )
        state["fruit_in_bin"] = self._reshape_bool(
            raw["fruit_in_bin"], (self.num_robots, self.num_locations)
        )
        state["fruits_unloaded"] = np.asarray(raw["fruits_unloaded"], dtype=bool)
        state["position_visited"] = np.asarray(raw["position_visited"], dtype=bool)
        state["robot_at"] = self._reshape_bool(raw["robot_at"], (self.num_robots, self.num_positions))
        if self._has_prev_position:
            state["prev_position"] = self._reshape_bool(
                raw["prev_position"], (self.num_robots, self.num_positions)
            )
        return state

    def bin_load(self, fruit_in_bin: np.ndarray) -> np.ndarray:
        return np.sum(fruit_in_bin.astype(float) * self.fruit_weights[np.newaxis, :], axis=1)

    def evaluate_reward(
        self, obs: Dict[str, Any], action: Dict[str, Any], next_obs: Dict[str, Any]
    ) -> float:
        robot_at = np.asarray(obs["robot_at"], dtype=bool)
        fruit_at = np.asarray(obs["fruit_at"], dtype=bool)
        fruit_collected = np.asarray(obs["fruit_collected"], dtype=bool)
        fruit_in_bin = np.asarray(obs["fruit_in_bin"], dtype=bool)
        position_visited = np.asarray(obs["position_visited"], dtype=bool)
        all_fruits_done = bool(obs["all_fruits_done"])

        navigate = np.asarray(action["navigate"], dtype=bool)
        grasp_fruit = np.asarray(action["grasp_fruit"], dtype=bool)
        load_to_bin = np.asarray(action["load_to_bin"], dtype=bool)
        unload = np.asarray(action["unload"], dtype=bool)
        wait = np.asarray(action["wait"], dtype=bool)

        reward = 0.0

        for robot_idx in range(self.num_robots):
            move_mask = (
                np.outer(robot_at[robot_idx], navigate[robot_idx]) & self.adjacent
            )
            reward -= float(np.sum(self.distance * move_mask)) * 0.03333

        reward -= float(np.sum(grasp_fruit) + np.sum(load_to_bin)) * 0.0666
        reward -= float(np.sum(unload)) * 0.2666

        current_load = self.bin_load(fruit_in_bin)

        for robot_idx in range(self.num_robots):
            if current_load[robot_idx] >= self.capacity_threshold:
                continue

            work_l = self.fruit_ripe & (
                (fruit_at & (~fruit_collected[robot_idx]))
                | (fruit_collected[robot_idx] & (~fruit_in_bin[robot_idx]))
            )
            exists_work_at_a1 = (self.reachable_from.T @ work_l.astype(int)) > 0
            abandon_mask = (
                np.outer(robot_at[robot_idx], navigate[robot_idx])
                & self.adjacent
                & np.outer(~self.unload_station, ~self.unload_station)
                & np.outer(exists_work_at_a1, np.ones(self.num_positions, dtype=bool))
            )
            reward -= float(np.sum(abandon_mask)) * 2.5

        if not all_fruits_done:
            reward -= float(np.sum(wait)) * 2.5
            reward -= 0.1666

        _ = position_visited
        _ = next_obs
        return float(reward)

    def step(self, obs: Dict[str, Any], action: Dict[str, Any]) -> Dict[str, Any]:
        next_obs = self.create_state_template()

        next_obs["all_fruits_done"] = bool(obs["all_fruits_done"])
        next_obs["fruit_at"] = np.asarray(obs["fruit_at"], dtype=bool).copy()
        next_obs["fruit_collected"] = np.asarray(obs["fruit_collected"], dtype=bool).copy()
        next_obs["fruit_in_bin"] = np.asarray(obs["fruit_in_bin"], dtype=bool).copy()
        next_obs["fruits_unloaded"] = np.asarray(obs["fruits_unloaded"], dtype=bool).copy()
        next_obs["position_visited"] = np.asarray(obs["position_visited"], dtype=bool).copy()
        next_obs["robot_at"] = np.asarray(obs["robot_at"], dtype=bool).copy()
        if self._has_prev_position:
            # prev_position' = robot_at (position before the transition)
            next_obs["prev_position"] = np.asarray(obs["robot_at"], dtype=bool).copy()

        navigate = np.asarray(action["navigate"], dtype=bool)
        grasp_fruit = np.asarray(action["grasp_fruit"], dtype=bool)
        load_to_bin = np.asarray(action["load_to_bin"], dtype=bool)
        unload = np.asarray(action["unload"], dtype=bool)

        # --- Navigate ---
        for robot_idx in range(self.num_robots):
            target_positions = np.where(navigate[robot_idx])[0]
            if target_positions.size > 0:
                next_obs["robot_at"][robot_idx] = False
                next_obs["robot_at"][robot_idx, target_positions[0]] = True

        # --- position_visited: set on arrival (any robot at the position) ---
        next_obs["position_visited"] |= np.any(next_obs["robot_at"], axis=0)

        # --- Grasp fruit (always succeeds per user assumption) ---
        for loc_idx in range(self.num_locations):
            grasp_success = False
            for robot_idx in range(self.num_robots):
                robot_pos = np.where(obs["robot_at"][robot_idx])[0]
                if robot_pos.size == 0:
                    continue
                pos_idx = robot_pos[0]
                if (
                    grasp_fruit[robot_idx, loc_idx]
                    and obs["fruit_at"][loc_idx]
                    and self.reachable_from[loc_idx, pos_idx]
                ):
                    grasp_success = True
                    next_obs["fruit_collected"][robot_idx, loc_idx] = True
            if grasp_success:
                next_obs["fruit_at"][loc_idx] = False

        # --- Unload and load_to_bin ---
        for robot_idx in range(self.num_robots):
            current_positions = np.where(obs["robot_at"][robot_idx])[0]
            current_pos = current_positions[0] if current_positions.size > 0 else None
            at_unload_station = (
                current_pos is not None and self.unload_station[current_pos]
            )

            for loc_idx in range(self.num_locations):
                if unload[robot_idx] and at_unload_station and obs["fruit_in_bin"][robot_idx, loc_idx]:
                    next_obs["fruit_in_bin"][robot_idx, loc_idx] = False
                    next_obs["fruit_collected"][robot_idx, loc_idx] = False
                    next_obs["fruits_unloaded"][loc_idx] = True
                    continue

                if (
                    load_to_bin[robot_idx, loc_idx]
                    and current_pos is not None
                    and self.reachable_from[loc_idx, current_pos]
                    and obs["fruit_collected"][robot_idx, loc_idx]
                    and not obs["fruit_in_bin"][robot_idx, loc_idx]
                ):
                    next_obs["fruit_in_bin"][robot_idx, loc_idx] = True

        next_obs["all_fruits_done"] = bool(
            np.all((~self.fruit_ripe) | next_obs["fruits_unloaded"])
        )
        return next_obs


if __name__ == "__main__":
    print("evaluator.py")
