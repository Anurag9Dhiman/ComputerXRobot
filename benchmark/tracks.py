"""Observation tracks: what an agent is allowed to see.

    state   Privileged low-dimensional state: proprioception, device and cursor positions,
            contact flags, and the task's next sub-goal (the character to type, or the widget
            to click, which mouse button, and where it is on screen). Separates *control*
            from perception and planning; an upper-bound setting for learning methods.

    vision  What a physical robot would have: images from the robot camera and the three
            device close-ups (robot_cam, keyboard_cam, mouse_cam, screen_cam), proprioception,
            a palm touch sensor, and the task instruction as text. No positions of devices,
            keys, cursor or widgets and no sub-goals: the agent has to read the keys and the
            screen, find the cursor, and plan the sequence of actions itself.

The harness removes everything a track does not allow before observations reach the agent.
Results are only comparable within a track.
"""

from __future__ import annotations

from dataclasses import dataclass

from environment.computer_env import EnvConfig

PROPRIOCEPTION = ("qpos", "qvel", "tool_pos", "tool_target", "finger")
VISION_CAMERAS = ("robot_cam", "keyboard_cam", "mouse_cam", "screen_cam")


@dataclass(frozen=True)
class Track:
    name: str
    description: str
    keys: tuple[str, ...]  # observation keys passed to the agent
    cameras: tuple[str, ...] = ()

    def env_config(self) -> EnvConfig:
        return EnvConfig(render_obs=bool(self.cameras), obs_cameras=self.cameras or ("robot_cam",))

    def filter(self, obs: dict) -> dict:
        return {k: obs[k] for k in self.keys if k in obs}


TRACKS: dict[str, Track] = {
    "state": Track(
        "state",
        "privileged state: proprioception, device / cursor positions and the next sub-goal",
        keys=PROPRIOCEPTION + ("instruction", "mouse_pos", "cursor", "key_down", "button_down", "palm_on_mouse",
                               "mouse_button", "keyboard_pos", "expected_char", "click_target", "click_button",
                               "click_target_px"),
    ),
    "vision": Track(
        "vision",
        "camera images + proprioception + palm touch + instruction text; no positions, no sub-goals",
        keys=PROPRIOCEPTION + ("instruction", "palm_on_mouse") + VISION_CAMERAS,
        cameras=VISION_CAMERAS,
    ),
}


def get_track(name: str) -> Track:
    if name not in TRACKS:
        raise ValueError(f"unknown track {name!r}; choose from {list(TRACKS)}")
    return TRACKS[name]
