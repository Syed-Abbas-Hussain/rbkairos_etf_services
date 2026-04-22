"""Tests for FruitHarvestingRewardEvaluator.step() in isolation.

These tests use the mock evaluator from fixtures to verify that the
step logic used during SIMULATION mode produces correct state transitions
for each action type: navigate, grasp_fruit, load_to_bin, unload, wait.

A separate integration block (skipped if pyRDDLGym is unavailable)
runs the real evaluator against the project's RDDL files.
"""

import numpy as np
import pytest

from fixtures import (
    make_mock_evaluator,
    make_initial_obs,
    make_action_template,
    POSITIONS, LOCATIONS, ROBOTS,
    NUM_ROBOTS, NUM_POSITIONS, NUM_LOCATIONS,
)


# ---------------------------------------------------------------------------
# Tests against the mock evaluator (always available)
# ---------------------------------------------------------------------------

class TestMockEvaluatorStep:
    """Verify the step() side_effect defined in fixtures.py."""

    def setup_method(self):
        self.ev = make_mock_evaluator()

    def _step(self, obs, action):
        return self.ev.step(obs, action)

    def test_navigate_moves_robot(self):
        obs = make_initial_obs()  # robot at a1 (idx 0)
        action = make_action_template()
        action["navigate"][0, 1] = True  # navigate to a2 (idx 1)
        next_obs = self._step(obs, action)
        assert next_obs["robot_at"][0, 1] is True or bool(next_obs["robot_at"][0, 1])
        assert not bool(next_obs["robot_at"][0, 0])

    def test_navigate_updates_only_moved_robot(self):
        obs = make_initial_obs()
        action = make_action_template()
        action["navigate"][0, 2] = True  # navigate to a3
        next_obs = self._step(obs, action)
        assert bool(next_obs["robot_at"][0, 2])
        # All other positions for robot0 must be False
        for p in range(NUM_POSITIONS):
            if p != 2:
                assert not bool(next_obs["robot_at"][0, p])

    def test_no_action_leaves_state_unchanged(self):
        obs = make_initial_obs()
        action = make_action_template()
        next_obs = self._step(obs, action)
        np.testing.assert_array_equal(next_obs["robot_at"], obs["robot_at"])
        np.testing.assert_array_equal(next_obs["fruit_at"], obs["fruit_at"])

    def test_state_is_deep_copy(self):
        obs = make_initial_obs()
        action = make_action_template()
        next_obs = self._step(obs, action)
        # Mutating next_obs must not affect obs
        next_obs["robot_at"][0, 0] = False
        assert bool(obs["robot_at"][0, 0])

    def test_fruits_unchanged_without_grasp_action(self):
        obs = make_initial_obs()
        action = make_action_template()
        next_obs = self._step(obs, action)
        np.testing.assert_array_equal(next_obs["fruit_at"], obs["fruit_at"])
        np.testing.assert_array_equal(next_obs["fruit_collected"], obs["fruit_collected"])

    def test_all_fruits_done_propagated(self):
        obs = make_initial_obs()
        obs["all_fruits_done"] = True
        action = make_action_template()
        next_obs = self._step(obs, action)
        assert next_obs["all_fruits_done"] is True


class TestMockEvaluatorReward:

    def setup_method(self):
        self.ev = make_mock_evaluator()

    def test_reward_is_float(self):
        obs = make_initial_obs()
        action = make_action_template()
        next_obs = self.ev.step(obs, action)
        reward = self.ev.evaluate_reward(obs, action, next_obs)
        assert isinstance(reward, (int, float))

    def test_evaluate_reward_called_with_correct_args(self):
        obs = make_initial_obs()
        action = make_action_template()
        next_obs = self.ev.step(obs, action)
        self.ev.evaluate_reward(obs, action, next_obs)
        self.ev.evaluate_reward.assert_called_once_with(obs, action, next_obs)


# ---------------------------------------------------------------------------
# Integration block — real pyRDDLGym + project RDDL files
# ---------------------------------------------------------------------------

import os

_DOMAIN_FILE = os.path.abspath(
    os.path.join(os.path.dirname(__file__),
                 "../../../problem_data/domain_new.rddl")
)
_INSTANCE_FILE = os.path.abspath(
    os.path.join(os.path.dirname(__file__),
                 "../../../problem_data/instance_new.rddl")
)

_RDDL_FILES_PRESENT = os.path.isfile(_DOMAIN_FILE) and os.path.isfile(_INSTANCE_FILE)

try:
    from pyRDDLGym.core.env import RDDLEnv as _RDDLEnv
    _PYRDLGYM_AVAILABLE = True
except ImportError:
    _PYRDLGYM_AVAILABLE = False

_INTEGRATION = _RDDL_FILES_PRESENT and _PYRDLGYM_AVAILABLE

_skip_integration = pytest.mark.skipif(
    not _INTEGRATION,
    reason="pyRDDLGym or project RDDL files not available",
)


@_skip_integration
class TestRealEvaluatorStep:
    """Integration tests using the actual FruitHarvestingRewardEvaluator."""

    @pytest.fixture(scope="class")
    def evaluator(self):
        import sys
        src_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../../src")
        )
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)
        from rbkairos_etf_services.evaluator import FruitHarvestingRewardEvaluator
        return FruitHarvestingRewardEvaluator(_DOMAIN_FILE, _INSTANCE_FILE)

    def test_initial_state_has_one_robot_position(self, evaluator):
        obs = evaluator.get_initial_state()
        for r in range(evaluator.num_robots):
            assert np.sum(obs["robot_at"][r]) == 1, (
                f"robot{r} must be at exactly one position"
            )

    def test_initial_state_fruits_present(self, evaluator):
        obs = evaluator.get_initial_state()
        assert np.any(obs["fruit_at"]), "At least one fruit must be present initially"

    def test_navigate_to_adjacent_position(self, evaluator):
        obs = evaluator.get_initial_state()
        current_pos = int(np.where(obs["robot_at"][0])[0][0])
        adjacent_targets = np.where(evaluator.adjacent[current_pos])[0]
        assert adjacent_targets.size > 0, "Starting position must have adjacent positions"
        target = int(adjacent_targets[0])

        action = evaluator.create_action_template()
        action["navigate"][0, target] = True
        next_obs = evaluator.step(obs, action)

        assert bool(next_obs["robot_at"][0, target]), "Robot should be at target after navigate"
        assert not bool(next_obs["robot_at"][0, current_pos]), "Robot should have left original position"

    def test_step_preserves_other_state(self, evaluator):
        obs = evaluator.get_initial_state()
        action = evaluator.create_action_template()  # no-op
        next_obs = evaluator.step(obs, action)
        np.testing.assert_array_equal(next_obs["fruit_at"], obs["fruit_at"])
        np.testing.assert_array_equal(next_obs["fruits_unloaded"], obs["fruits_unloaded"])

    def test_grasp_fruit_at_reachable_location(self, evaluator):
        obs = evaluator.get_initial_state()
        current_pos = int(np.where(obs["robot_at"][0])[0][0])
        reachable = np.where(evaluator.reachable_from[:, current_pos])[0]
        ripe_reachable = [l for l in reachable if evaluator.fruit_ripe[l] and obs["fruit_at"][l]]
        if not ripe_reachable:
            pytest.skip("No ripe reachable fruit from starting position")

        loc_idx = ripe_reachable[0]
        action = evaluator.create_action_template()
        action["grasp_fruit"][0, loc_idx] = True
        next_obs = evaluator.step(obs, action)

        assert bool(next_obs["fruit_collected"][0, loc_idx])
        assert not bool(next_obs["fruit_at"][loc_idx])

    def test_load_to_bin_after_grasp(self, evaluator):
        obs = evaluator.get_initial_state()
        current_pos = int(np.where(obs["robot_at"][0])[0][0])
        reachable = np.where(evaluator.reachable_from[:, current_pos])[0]
        ripe_reachable = [l for l in reachable if evaluator.fruit_ripe[l] and obs["fruit_at"][l]]
        if not ripe_reachable:
            pytest.skip("No ripe reachable fruit from starting position")

        loc_idx = ripe_reachable[0]

        # Grasp first
        action1 = evaluator.create_action_template()
        action1["grasp_fruit"][0, loc_idx] = True
        obs2 = evaluator.step(obs, action1)

        # Then load
        action2 = evaluator.create_action_template()
        action2["load_to_bin"][0, loc_idx] = True
        obs3 = evaluator.step(obs2, action2)

        assert bool(obs3["fruit_in_bin"][0, loc_idx])

    def test_evaluate_reward_returns_float(self, evaluator):
        obs = evaluator.get_initial_state()
        action = evaluator.create_action_template()
        next_obs = evaluator.step(obs, action)
        reward = evaluator.evaluate_reward(obs, action, next_obs)
        assert isinstance(reward, float)

    def test_unload_at_unload_station_clears_bin(self, evaluator):
        obs = evaluator.get_initial_state()

        # Teleport robot to an unload station
        unload_positions = np.where(evaluator.unload_station)[0]
        assert unload_positions.size > 0
        unload_pos = int(unload_positions[0])
        obs["robot_at"][0] = False
        obs["robot_at"][0, unload_pos] = True

        # Put fruit in bin
        obs["fruit_in_bin"][0, 0] = True
        obs["fruit_collected"][0, 0] = True

        action = evaluator.create_action_template()
        action["unload"][0] = True
        next_obs = evaluator.step(obs, action)

        assert not bool(next_obs["fruit_in_bin"][0, 0])
        assert bool(next_obs["fruits_unloaded"][0])

    def test_all_fruits_done_when_all_unloaded(self, evaluator):
        obs = evaluator.get_initial_state()
        unload_positions = np.where(evaluator.unload_station)[0]
        unload_pos = int(unload_positions[0])
        obs["robot_at"][0] = False
        obs["robot_at"][0, unload_pos] = True

        # Mark all ripe fruit as in-bin and collected
        ripe_indices = np.where(evaluator.fruit_ripe)[0]
        obs["fruit_at"] = np.zeros(evaluator.num_locations, dtype=bool)
        obs["fruit_collected"][0, ripe_indices] = True
        obs["fruit_in_bin"][0, ripe_indices] = True

        action = evaluator.create_action_template()
        action["unload"][0] = True
        next_obs = evaluator.step(obs, action)

        assert bool(next_obs["all_fruits_done"])
