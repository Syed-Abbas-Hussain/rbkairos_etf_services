"""Tests for ServiceCaller.parse_action in SIMULATION mode.

parse_action maps (action_name, action_params) to
(robot_idx, aisle_index, location_index).
"""

import pytest

from fixtures import (
    make_caller,
    POSITIONS, LOCATIONS, ROBOTS,
    NUM_POSITIONS, NUM_LOCATIONS,
)


class TestParseAction:

    def setup_method(self):
        self.caller = make_caller()

    # ------------------------------------------------------------------
    # navigate
    # ------------------------------------------------------------------

    def test_navigate_known_robot_and_position(self):
        robot_idx, aisle_idx, loc_idx = self.caller.parse_action(
            "navigate", ["robot1", "a2"]
        )
        assert robot_idx == 0
        assert aisle_idx == POSITIONS.index("a2")
        assert loc_idx == -1

    def test_navigate_first_position(self):
        _, aisle_idx, _ = self.caller.parse_action("navigate", ["robot1", "a1"])
        assert aisle_idx == 0

    def test_navigate_unload_station(self):
        _, aisle_idx, _ = self.caller.parse_action("navigate", ["robot1", "unload1"])
        assert aisle_idx == POSITIONS.index("unload1")

    def test_navigate_unknown_position_returns_minus_one(self):
        _, aisle_idx, _ = self.caller.parse_action("navigate", ["robot1", "a99"])
        assert aisle_idx == -1

    def test_navigate_missing_position_param(self):
        _, aisle_idx, _ = self.caller.parse_action("navigate", ["robot1"])
        assert aisle_idx == -1

    # ------------------------------------------------------------------
    # grasp_fruit
    # ------------------------------------------------------------------

    def test_grasp_fruit_known_robot_and_location(self):
        robot_idx, aisle_idx, loc_idx = self.caller.parse_action(
            "grasp_fruit", ["robot1", "l1"]
        )
        assert robot_idx == 0
        assert aisle_idx == -1
        assert loc_idx == LOCATIONS.index("l1")

    def test_grasp_fruit_last_location(self):
        last = LOCATIONS[-1]
        _, _, loc_idx = self.caller.parse_action("grasp_fruit", ["robot1", last])
        assert loc_idx == len(LOCATIONS) - 1

    def test_grasp_fruit_unknown_location_returns_minus_one(self):
        _, _, loc_idx = self.caller.parse_action("grasp_fruit", ["robot1", "l99"])
        assert loc_idx == -1

    # ------------------------------------------------------------------
    # load_to_bin
    # ------------------------------------------------------------------

    def test_load_to_bin_returns_location_index(self):
        robot_idx, aisle_idx, loc_idx = self.caller.parse_action(
            "load_to_bin", ["robot1", "l2"]
        )
        assert robot_idx == 0
        assert aisle_idx == -1
        assert loc_idx == LOCATIONS.index("l2")

    # ------------------------------------------------------------------
    # unload / wait / NOOP — no second param expected
    # ------------------------------------------------------------------

    def test_unload_no_indices(self):
        robot_idx, aisle_idx, loc_idx = self.caller.parse_action(
            "unload", ["robot1"]
        )
        assert robot_idx == 0
        assert aisle_idx == -1
        assert loc_idx == -1

    def test_wait_no_indices(self):
        robot_idx, aisle_idx, loc_idx = self.caller.parse_action(
            "wait", ["robot1"]
        )
        assert robot_idx == 0
        assert aisle_idx == -1
        assert loc_idx == -1

    def test_noop_empty_params(self):
        robot_idx, aisle_idx, loc_idx = self.caller.parse_action("NOOP", [])
        assert robot_idx == self.caller.default_robot_idx
        assert aisle_idx == -1
        assert loc_idx == -1

    # ------------------------------------------------------------------
    # Robot resolution
    # ------------------------------------------------------------------

    def test_unknown_robot_falls_back_to_default(self):
        robot_idx, _, _ = self.caller.parse_action("navigate", ["ghost_bot", "a1"])
        assert robot_idx == self.caller.default_robot_idx

    def test_empty_params_uses_default_robot(self):
        robot_idx, _, _ = self.caller.parse_action("wait", [])
        assert robot_idx == self.caller.default_robot_idx
