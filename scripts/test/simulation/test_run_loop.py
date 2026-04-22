"""Tests for ServiceCaller.run() in SIMULATION mode.

The planner is mocked via caller.submit_obs.  Each test scripts
a sequence of planner responses, then verifies:
  - the loop exits cleanly on terminal actions (ROUND_END / FAILED / ERROR)
  - the evaluator.step() is called for every non-terminal action
  - observations are submitted to the planner on every iteration
  - state advances correctly through the sequence
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, call

from fixtures import (
    make_caller,
    make_initial_obs,
    POSITIONS, LOCATIONS, ROBOTS,
    NUM_ROBOTS, NUM_POSITIONS, NUM_LOCATIONS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _planner_response(action_name, *params):
    resp = MagicMock()
    resp.action_name = action_name
    resp.action_params = list(params)
    return resp


def _script_planner(caller, responses):
    """Wire caller.submit_obs to return responses in order."""
    caller.submit_obs.side_effect = [_planner_response(*r) for r in responses]


# ---------------------------------------------------------------------------
# Terminal-action tests
# ---------------------------------------------------------------------------

class TestTerminalActions:

    @pytest.mark.parametrize("terminal", ["ROUND_END", "FAILED", "ERROR"])
    def test_loop_stops_on_terminal(self, terminal):
        caller = make_caller()
        _script_planner(caller, [(terminal,)])
        caller.run()
        assert caller.submit_obs.call_count == 1

    def test_round_end_after_one_navigate(self):
        caller = make_caller()
        _script_planner(caller, [
            ("navigate", "robot1", "a2"),
            ("ROUND_END",),
        ])
        caller.run()
        assert caller.submit_obs.call_count == 2

    def test_evaluator_step_called_for_real_actions(self):
        caller = make_caller()
        _script_planner(caller, [
            ("navigate", "robot1", "a2"),
            ("navigate", "robot1", "a3"),
            ("ROUND_END",),
        ])
        caller.run()
        assert caller.evaluator.step.call_count == 2

    def test_evaluator_step_not_called_for_terminal(self):
        caller = make_caller()
        _script_planner(caller, [("FAILED",)])
        caller.run()
        caller.evaluator.step.assert_not_called()


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

class TestNavigateInSimulation:

    def test_robot_position_updates_after_navigate(self):
        caller = make_caller()
        _script_planner(caller, [
            ("navigate", "robot1", "a2"),
            ("ROUND_END",),
        ])
        caller.run()
        # robot_at should reflect the new position (a2 = index 1)
        assert caller.obs["robot_at"][0, 1] is np.bool_(True) or bool(caller.obs["robot_at"][0, 1])

    def test_navigate_unknown_position_does_not_crash(self):
        caller = make_caller()
        _script_planner(caller, [
            ("navigate", "robot1", "a_does_not_exist"),
            ("ROUND_END",),
        ])
        caller.run()  # must not raise

    def test_navigate_to_each_position(self):
        for pos in POSITIONS:
            caller = make_caller()
            _script_planner(caller, [
                ("navigate", "robot1", pos),
                ("ROUND_END",),
            ])
            caller.run()


# ---------------------------------------------------------------------------
# Grasp / load / unload / wait / NOOP
# ---------------------------------------------------------------------------

class TestAllActionTypes:

    def _run_single(self, action_name, *params):
        caller = make_caller()
        _script_planner(caller, [
            (action_name, *params),
            ("ROUND_END",),
        ])
        caller.run()
        return caller

    def test_grasp_fruit_does_not_crash(self):
        self._run_single("grasp_fruit", "robot1", "l1")

    def test_load_to_bin_does_not_crash(self):
        self._run_single("load_to_bin", "robot1", "l1")

    def test_unload_does_not_crash(self):
        self._run_single("unload", "robot1")

    def test_wait_does_not_crash(self):
        self._run_single("wait", "robot1")

    def test_noop_does_not_crash(self):
        self._run_single("NOOP")

    def test_unknown_action_does_not_crash(self):
        self._run_single("unknown_action_xyz")

    @pytest.mark.parametrize("action", [
        ("navigate", "robot1", "a2"),
        ("grasp_fruit", "robot1", "l1"),
        ("load_to_bin", "robot1", "l2"),
        ("unload", "robot1"),
        ("wait", "robot1"),
        ("NOOP",),
    ])
    def test_step_called_once(self, action):
        caller = make_caller()
        _script_planner(caller, [action, ("ROUND_END",)])
        caller.run()
        assert caller.evaluator.step.call_count == 1


# ---------------------------------------------------------------------------
# Observation submission
# ---------------------------------------------------------------------------

class TestObservationSubmission:

    def test_submit_obs_called_each_iteration(self):
        caller = make_caller()
        _script_planner(caller, [
            ("navigate", "robot1", "a2"),
            ("navigate", "robot1", "a3"),
            ("navigate", "robot1", "a4"),
            ("ROUND_END",),
        ])
        caller.run()
        assert caller.submit_obs.call_count == 4

    def test_reward_passed_to_submit_obs(self):
        caller = make_caller()
        caller.evaluator.evaluate_reward.return_value = -5.0
        _script_planner(caller, [
            ("navigate", "robot1", "a2"),
            ("ROUND_END",),
        ])
        caller.run()
        # Second call to submit_obs should carry the reward from the first step
        second_call_args = caller.submit_obs.call_args_list[1]
        reward_arg = second_call_args[0][1]  # positional arg index 1
        assert reward_arg == pytest.approx(-5.0)


# ---------------------------------------------------------------------------
# Idle-count termination
# ---------------------------------------------------------------------------

class TestIdleCountTermination:

    def test_loop_breaks_at_idle_limit(self):
        caller = make_caller()
        # Return navigate forever — loop must exit at idle_count > 200
        caller.submit_obs.return_value = _planner_response("navigate", "robot1", "a2")
        caller.run()
        assert caller.idle_count > 200

    def test_loop_breaks_when_all_fruits_done(self):
        caller = make_caller()

        step_call_count = [0]

        def _step_done(obs, action):
            step_call_count[0] += 1
            next_obs = {k: v.copy() if isinstance(v, np.ndarray) else v
                        for k, v in obs.items()}
            if step_call_count[0] >= 3:
                next_obs["all_fruits_done"] = True
            return next_obs

        caller.evaluator.step.side_effect = _step_done
        caller.submit_obs.return_value = _planner_response("navigate", "robot1", "a2")
        caller.run()
        assert caller.obs["all_fruits_done"] is True


# ---------------------------------------------------------------------------
# Action-index boundary
# ---------------------------------------------------------------------------

class TestActionIndexBoundaries:

    def test_navigate_out_of_range_index_does_not_crash(self):
        """aisle_index == -1 means no motion entry is fetched; loop continues."""
        caller = make_caller()
        _script_planner(caller, [
            ("navigate", "robot1", "nonexistent_pos"),
            ("ROUND_END",),
        ])
        caller.run()

    def test_grasp_out_of_range_index_does_not_crash(self):
        caller = make_caller()
        _script_planner(caller, [
            ("grasp_fruit", "robot1", "nonexistent_loc"),
            ("ROUND_END",),
        ])
        caller.run()
