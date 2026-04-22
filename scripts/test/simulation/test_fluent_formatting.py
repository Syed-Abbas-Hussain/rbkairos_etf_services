"""Tests for ServiceCaller fluent-to-KeyValue formatting helpers.

Verifies that append_scalar_fluent, append_vector_fluent,
append_robot_fluent, and append_robot_scalar_fluent produce
correctly named KeyValue pairs with "true"/"false" string values.
"""

import numpy as np
import pytest

from fixtures import make_caller, POSITIONS, LOCATIONS, ROBOTS, NUM_POSITIONS, NUM_LOCATIONS


def kv_dict(kv_list):
    """Convert a list of KeyValue-like objects to {key: value} dict."""
    return {kv.key: kv.value for kv in kv_list}


class TestAppendScalarFluent:

    def setup_method(self):
        self.caller = make_caller()

    def test_true_value(self):
        buf = []
        self.caller.append_scalar_fluent(buf, "all_fruits_done", True)
        assert len(buf) == 1
        assert buf[0].key == "all_fruits_done"
        assert buf[0].value == "true"

    def test_false_value(self):
        buf = []
        self.caller.append_scalar_fluent(buf, "all_fruits_done", False)
        assert buf[0].value == "false"

    def test_truthy_int(self):
        buf = []
        self.caller.append_scalar_fluent(buf, "flag", 1)
        assert buf[0].value == "true"

    def test_falsy_zero(self):
        buf = []
        self.caller.append_scalar_fluent(buf, "flag", 0)
        assert buf[0].value == "false"

    def test_numpy_true(self):
        buf = []
        self.caller.append_scalar_fluent(buf, "flag", np.bool_(True))
        assert buf[0].value == "true"


class TestAppendVectorFluent:

    def setup_method(self):
        self.caller = make_caller()

    def test_length_matches_objects(self):
        buf = []
        values = np.array([True, False, True, False])
        self.caller.append_vector_fluent(buf, "fruit_at", values, LOCATIONS)
        assert len(buf) == len(LOCATIONS)

    def test_key_names_include_object(self):
        buf = []
        values = np.zeros(len(LOCATIONS), dtype=bool)
        self.caller.append_vector_fluent(buf, "fruit_at", values, LOCATIONS)
        for kv, loc in zip(buf, LOCATIONS):
            assert kv.key == f"fruit_at({loc})"

    def test_all_false(self):
        buf = []
        values = np.zeros(len(LOCATIONS), dtype=bool)
        self.caller.append_vector_fluent(buf, "fruit_at", values, LOCATIONS)
        assert all(kv.value == "false" for kv in buf)

    def test_all_true(self):
        buf = []
        values = np.ones(len(LOCATIONS), dtype=bool)
        self.caller.append_vector_fluent(buf, "fruit_at", values, LOCATIONS)
        assert all(kv.value == "true" for kv in buf)

    def test_mixed_values(self):
        buf = []
        values = np.array([True, False, True, False])
        self.caller.append_vector_fluent(buf, "fruit_at", values, LOCATIONS)
        d = kv_dict(buf)
        assert d["fruit_at(l1)"] == "true"
        assert d["fruit_at(l2)"] == "false"
        assert d["fruit_at(l3)"] == "true"
        assert d["fruit_at(l4)"] == "false"

    def test_position_visited(self):
        buf = []
        values = np.array([True, False, False, False, False])
        self.caller.append_vector_fluent(buf, "position_visited", values, POSITIONS)
        d = kv_dict(buf)
        assert d["position_visited(a1)"] == "true"
        assert d["position_visited(a2)"] == "false"


class TestAppendRobotFluent:

    def setup_method(self):
        self.caller = make_caller()

    def test_produces_robot_cross_object_keys(self):
        buf = []
        values = np.zeros((1, len(LOCATIONS)), dtype=bool)
        self.caller.append_robot_fluent(buf, "fruit_collected", values, LOCATIONS)
        assert len(buf) == len(ROBOTS) * len(LOCATIONS)

    def test_key_format(self):
        buf = []
        values = np.zeros((1, len(LOCATIONS)), dtype=bool)
        self.caller.append_robot_fluent(buf, "fruit_collected", values, LOCATIONS)
        for kv in buf:
            assert kv.key.startswith("fruit_collected(robot1,")
            assert kv.key.endswith(")")

    def test_true_entry(self):
        buf = []
        values = np.zeros((1, len(LOCATIONS)), dtype=bool)
        values[0, 2] = True  # robot1, l3
        self.caller.append_robot_fluent(buf, "fruit_in_bin", values, LOCATIONS)
        d = kv_dict(buf)
        assert d["fruit_in_bin(robot1,l3)"] == "true"
        assert d["fruit_in_bin(robot1,l1)"] == "false"

    def test_robot_at_matrix(self):
        buf = []
        robot_at = np.zeros((1, len(POSITIONS)), dtype=bool)
        robot_at[0, 0] = True  # robot1 at a1
        self.caller.append_robot_fluent(buf, "robot_at", robot_at, POSITIONS)
        d = kv_dict(buf)
        assert d["robot_at(robot1,a1)"] == "true"
        for pos in POSITIONS[1:]:
            assert d[f"robot_at(robot1,{pos})"] == "false"


class TestAppendRobotScalarFluent:

    def setup_method(self):
        self.caller = make_caller()

    def test_one_key_per_robot(self):
        buf = []
        values = np.zeros(len(ROBOTS), dtype=bool)
        self.caller.append_robot_scalar_fluent(buf, "unload", values)
        assert len(buf) == len(ROBOTS)

    def test_key_format(self):
        buf = []
        values = np.array([True])
        self.caller.append_robot_scalar_fluent(buf, "unload", values)
        assert buf[0].key == "unload(robot1)"
        assert buf[0].value == "true"

    def test_false_value(self):
        buf = []
        values = np.array([False])
        self.caller.append_robot_scalar_fluent(buf, "wait", values)
        assert buf[0].value == "false"
