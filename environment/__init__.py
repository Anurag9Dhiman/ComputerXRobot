"""Simulation environment: MuJoCo scene, simulated desktop, physical-to-UI mapping, tasks."""

import os
import sys

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
if sys.platform == "darwin":  # CGL renders offscreen without a window (must precede mujoco import).
    os.environ.setdefault("MUJOCO_GL", "cgl")
