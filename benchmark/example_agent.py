"""Template for a new vision-track agent.

    python -m benchmark run --agent benchmark.example_agent:ExampleVisionAgent --all-test --workers 8

Copy this file, keep the class attributes and the two methods, and replace the body of `act`.
"""

from __future__ import annotations

import numpy as np

from benchmark.agents import Agent


class ExampleVisionAgent(Agent):
    name = "example-vision"  # used in result tables
    track = "vision"  # what the harness lets this agent see (benchmark/tracks.py)

    def reset(self, instruction: str, seed: int) -> None:
        # e.g. 'Replace the draft in the editor with "send report"'
        self.instruction = instruction
        self.rng = np.random.default_rng(seed)

    def act(self, obs: dict) -> np.ndarray:
        # obs (vision track):
        #   robot_cam (240, 320, 3), keyboard_cam (256, 512, 3), mouse_cam (320, 320, 3),
        #   screen_cam (400, 640, 3)    uint8 RGB camera images
        #   qpos, qvel (6,)             arm joints + finger
        #   tool_pos, tool_target (3,)  hand position and its commanded target (m)
        #   finger                      finger extension in [0, 1]
        #   palm_on_mouse               palm touch sensor
        #   instruction                 the task in words
        # Return [dx, dy, dz, finger]: target change in metres (|d| <= 0.006 per step) and the
        # finger command in [0, 1]. This placeholder holds still.
        return np.zeros(4)
