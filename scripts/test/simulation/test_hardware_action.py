"""Tests for ServiceCaller._execute_hardware_action in SIMULATION mode.

In SIMULATION mode the method must return True for every action type,
regardless of arguments.  No hardware service is ever called.
"""

import numpy as np
import pytest

from fixtures import make_caller, NUM_POSITIONS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ZERO_ACTION = np.zeros(6, dtype=float)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestExecuteHardwareActionSimulation:

    def setup_method(self):
        self.caller = make_caller()

    def test_mode_is_simulation(self):
        assert self.caller.mode == "simulation"

    def test_navigate_returns_true(self):
        result = self.caller._execute_hardware_action(0, "navigate", ZERO_ACTION)
        assert result is True

    def test_grasp_fruit_returns_true(self):
        result = self.caller._execute_hardware_action(0, "grasp_fruit", ZERO_ACTION)
        assert result is True

    def test_load_to_bin_returns_true(self):
        result = self.caller._execute_hardware_action(0, "load_to_bin", ZERO_ACTION)
        assert result is True

    def test_unload_returns_true(self):
        result = self.caller._execute_hardware_action(0, "unload", ZERO_ACTION)
        assert result is True

    def test_wait_returns_true(self):
        result = self.caller._execute_hardware_action(0, "wait", ZERO_ACTION)
        assert result is True

    def test_noop_returns_true(self):
        result = self.caller._execute_hardware_action(0, "NOOP", ZERO_ACTION)
        assert result is True

    def test_unknown_action_returns_true(self):
        result = self.caller._execute_hardware_action(0, "unknown_action_xyz", ZERO_ACTION)
        assert result is True

    def test_any_robot_idx_returns_true(self):
        for robot_idx in range(3):
            assert self.caller._execute_hardware_action(robot_idx, "navigate", ZERO_ACTION) is True

    def test_no_hardware_service_called(self):
        self.caller._execute_hardware_action(0, "navigate", ZERO_ACTION)
        assert self.caller.perform_action is None
        assert self.caller.robot_action_servers == []

    @pytest.mark.parametrize("action_name", [
        "navigate", "grasp_fruit", "load_to_bin", "unload", "wait", "NOOP",
    ])
    def test_all_actions_parametrized(self, action_name):
        real_action = np.random.rand(6)
        assert self.caller._execute_hardware_action(0, action_name, real_action) is True
