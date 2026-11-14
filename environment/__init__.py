"""Simulation environment: MuJoCo scene, simulated desktop, physical-to-UI mapping, tasks."""

import os
import sys

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
# Offscreen rendering backends; both must be chosen before mujoco is imported.
if sys.platform == "darwin":  # CGL renders offscreen without a window.
    os.environ.setdefault("MUJOCO_GL", "cgl")
elif sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
    os.environ.setdefault("MUJOCO_GL", "egl")  # headless server: EGL on the GPU driver
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # pygame without a display
