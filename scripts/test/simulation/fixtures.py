"""Shared fixtures for SIMULATION mode tests.

ServiceCaller is constructed via __new__ so no ROS master is needed.
All ROS module stubs are injected into sys.modules before service_caller
is first imported; subsequent imports reuse the cached module.
"""

import json
import os
import sys
import tempfile
from unittest.mock import MagicMock

import numpy as np

# ---------------------------------------------------------------------------
# Tiny domain description used by every test
# ---------------------------------------------------------------------------
POSITIONS = ["a1", "a2", "a3", "a4", "unload1"]
LOCATIONS = ["l1", "l2", "l3", "l4"]
ROBOTS = ["robot1"]

NUM_ROBOTS = len(ROBOTS)
NUM_POSITIONS = len(POSITIONS)
NUM_LOCATIONS = len(LOCATIONS)


# ---------------------------------------------------------------------------
# Inject ROS stubs once, before service_caller is imported
# ---------------------------------------------------------------------------

def _stub_ros_modules():
    """Inject lightweight stubs for every ROS module used by service_caller."""
    if "rospy" in sys.modules:
        return  # already done

    key_value_instances = []

    class _KeyValue:
        def __init__(self, key, value):
            self.key = key
            self.value = value
            key_value_instances.append(self)

        def __repr__(self):
            return f"KeyValue({self.key!r}, {self.value!r})"

    prost_ros_msg = MagicMock()
    prost_ros_msg.KeyValue = _KeyValue

    sys.modules.setdefault("rospy", MagicMock())
    sys.modules.setdefault("rospkg", MagicMock())
    sys.modules.setdefault("prost_ros", MagicMock())
    sys.modules.setdefault("prost_ros.msg", prost_ros_msg)
    sys.modules.setdefault("prost_ros.srv", MagicMock())
    sys.modules.setdefault("rbkairos_etf_services", MagicMock())
    sys.modules.setdefault("rbkairos_etf_services.srv", MagicMock())
    sys.modules.setdefault("rbkairos_etf_services.evaluator", MagicMock())


_stub_ros_modules()

# Now it is safe to import the module under test
_scripts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

import service_caller as _sc_mod  # noqa: E402  (import after sys.modules patching)


# ---------------------------------------------------------------------------
# State / action helpers
# ---------------------------------------------------------------------------

def make_initial_obs():
    obs = {
        "all_fruits_done": False,
        "fruit_at": np.ones(NUM_LOCATIONS, dtype=bool),
        "fruit_collected": np.zeros((NUM_ROBOTS, NUM_LOCATIONS), dtype=bool),
        "fruit_in_bin": np.zeros((NUM_ROBOTS, NUM_LOCATIONS), dtype=bool),
        "fruits_unloaded": np.zeros(NUM_LOCATIONS, dtype=bool),
        "position_visited": np.zeros(NUM_POSITIONS, dtype=bool),
        "robot_at": np.zeros((NUM_ROBOTS, NUM_POSITIONS), dtype=bool),
    }
    obs["robot_at"][0, 0] = True  # robot1 starts at a1 (index 0)
    return obs


def make_action_template():
    return {
        "grasp_fruit": np.zeros((NUM_ROBOTS, NUM_LOCATIONS), dtype=bool),
        "load_to_bin": np.zeros((NUM_ROBOTS, NUM_LOCATIONS), dtype=bool),
        "navigate": np.zeros((NUM_ROBOTS, NUM_POSITIONS), dtype=bool),
        "unload": np.zeros(NUM_ROBOTS, dtype=bool),
        "wait": np.zeros(NUM_ROBOTS, dtype=bool),
    }


def make_mock_evaluator():
    ev = MagicMock()
    ev.positions = list(POSITIONS)
    ev.locations = list(LOCATIONS)
    ev.robots = list(ROBOTS)
    ev.robot_to_idx = {"robot1": 0}
    # unload1 is the last position (index 4)
    ev.unload_station = np.array([False, False, False, False, True])
    ev.get_initial_state.return_value = make_initial_obs()
    ev.create_action_template.side_effect = make_action_template

    def _step(obs, action):
        next_obs = {k: v.copy() if isinstance(v, np.ndarray) else v
                    for k, v in obs.items()}
        nav = np.asarray(action["navigate"], dtype=bool)
        for r in range(NUM_ROBOTS):
            targets = np.where(nav[r])[0]
            if targets.size > 0:
                next_obs["robot_at"][r] = False
                next_obs["robot_at"][r, targets[0]] = True
        return next_obs

    ev.step.side_effect = _step
    ev.evaluate_reward.return_value = -0.1666
    return ev


def make_actions_file(tmp_dir: str) -> str:
    """Write a minimal actions.json and return its path."""
    data = {
        "navigate": [[float(i), 0.0, 0.0, 0.0, 0.0, 0.0] for i in range(NUM_POSITIONS)],
        "grasp_fruit": [[0.0, 0.0, float(j) * 0.1, 0.0, 0.0, 0.0] for j in range(NUM_LOCATIONS)],
        "load_to_bin": [[-0.3, 0.2, 0.35, 0.0, 0.0, 0.0]],
        "unload": [[0.0, 0.0, 0.5, 0.0, 0.0, 0.0]],
    }
    path = os.path.join(tmp_dir, "actions.json")
    with open(path, "w") as f:
        json.dump(data, f)
    return path


# ---------------------------------------------------------------------------
# ServiceCaller factory — bypasses __init__ entirely
# ---------------------------------------------------------------------------

def make_caller(evaluator=None, actions_file=None, tmp_dir=None) -> _sc_mod.ServiceCaller:
    """Return a fully-configured ServiceCaller in SIMULATION mode."""
    if evaluator is None:
        evaluator = make_mock_evaluator()
    if tmp_dir is None:
        tmp_dir = tempfile.mkdtemp()
    if actions_file is None:
        actions_file = make_actions_file(tmp_dir)

    caller = _sc_mod.ServiceCaller.__new__(_sc_mod.ServiceCaller)
    caller.mode = _sc_mod.ServiceCaller.MODE_SIMULATION
    caller.perform_action = None
    caller.robot_action_servers = []
    caller.evaluator = evaluator
    caller.obs = evaluator.get_initial_state()
    caller.reward = 0.0
    caller.idle_count = 0
    caller.robot_names = list(ROBOTS)
    caller.positions = list(POSITIONS)
    caller.locations = list(LOCATIONS)
    caller.robot_to_idx = {"robot1": 0}
    caller.pos_to_idx = {p: i for i, p in enumerate(POSITIONS)}
    caller.loc_to_idx = {l: i for i, l in enumerate(LOCATIONS)}
    caller.default_robot_idx = 0
    caller.start_planning = MagicMock(return_value=MagicMock(success=True))
    caller.submit_obs = MagicMock()

    with open(actions_file) as f:
        caller.ACTION_DATA = json.load(f)

    return caller
