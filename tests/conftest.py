import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import environment  # noqa: E402,F401  (sets MUJOCO_GL / pygame env vars)
from environment.computer_env import ComputerUseEnv, EnvConfig  # noqa: E402


@pytest.fixture(scope="session")
def env():
    """One headless environment shared by the tests (model compilation is the slow part)."""
    e = ComputerUseEnv(EnvConfig(render_obs=False))
    yield e
    e.close()
