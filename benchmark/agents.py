"""The agent interface and the built-in baselines.

An agent implements two methods:

    reset(instruction: str, seed: int) -> None     start of an episode
    act(obs: dict) -> np.ndarray                    one action per control step (50 Hz)

The action is (4,) float: [dx, dy, dz, finger] -- the change of the hand's target position in
metres (each component clipped to +-0.006, i.e. 0.3 m/s) and the finger extension command in
[0, 1]. `obs` holds only what the agent's track allows (benchmark/tracks.py).

Agents are named by a spec string:

    random                  uniform random actions (lower bound)
    oracle                  scripted expert with privileged simulator access (upper bound;
                            reported separately, not a track entry)
    adaptive                the same expert without privileged access: it identifies the
                            hidden dynamics online from state-track observations and its own
                            commands (learning/adaptive.py)
    fixed                   the same, assuming nominal dynamics instead of identifying them
                            (control for what identification contributes)
    bc:<checkpoint.pt>      behaviour cloning policy (learning/policy.py), state track
    vbc:<checkpoint.pt>     behaviour cloning from pixels (learning/vision.py), vision track;
                            checkpoints trained with --subgoal need --track vision_subgoal
    <module>:<Class>        your own agent class; constructed as Class(env) and must define
                            `name` and `track` class attributes
"""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np

from environment.computer_env import ComputerUseEnv


class Agent:
    name = "agent"
    track = "state"
    privileged = False  # True if it reads simulator state beyond its track

    def __init__(self, env: ComputerUseEnv):
        self.max_step = env.config.max_step

    def reset(self, instruction: str, seed: int) -> None:
        pass

    def act(self, obs: dict) -> np.ndarray:
        raise NotImplementedError


class RandomAgent(Agent):
    name = "random"
    track = "state"

    def reset(self, instruction, seed):
        self.rng = np.random.default_rng(seed)

    def act(self, obs):
        return np.array([*self.rng.uniform(-self.max_step, self.max_step, 3), self.rng.uniform(0.0, 1.0)])


class OracleAgent(Agent):
    name = "oracle"
    track = "state"
    privileged = True

    def __init__(self, env):
        super().__init__(env)
        from learning.scripted_policy import ScriptedPolicy

        self.policy = ScriptedPolicy(env)

    def reset(self, instruction, seed):
        self.policy.reset()

    def act(self, obs):
        return self.policy.act()


class AdaptiveAgent(Agent):
    name = "adaptive-expert"
    track = "state"

    def __init__(self, env):
        super().__init__(env)
        from learning.adaptive import AdaptiveExpert

        self.policy = AdaptiveExpert(env.config)  # public configuration only, never env state

    def reset(self, instruction, seed):
        self.policy.reset()

    def act(self, obs):
        return self.policy.act(obs)


class FixedAgent(AdaptiveAgent):
    name = "fixed-expert"

    def __init__(self, env):
        Agent.__init__(self, env)
        from learning.adaptive import AdaptiveExpert

        self.policy = AdaptiveExpert(env.config, identify=False)


class BCAgent(Agent):
    name = "bc"
    track = "state"

    def __init__(self, env, checkpoint: str):
        super().__init__(env)
        from learning.policy import BCPolicy

        self.policy = BCPolicy(checkpoint, env.config.max_step, device="cpu")
        self.checkpoint = checkpoint

    def reset(self, instruction, seed):
        self.policy.reset()

    def act(self, obs):
        return self.policy.act(obs)


class VisionBCAgent(Agent):
    name = "vision-bc"
    track = "vision"

    def __init__(self, env, checkpoint: str):
        super().__init__(env)
        import torch

        from learning.vision import VisionBCPolicy

        torch.set_num_threads(1)  # one evaluation worker per core
        self.policy = VisionBCPolicy(checkpoint, env.config.max_step, device="cpu")
        self.checkpoint = checkpoint

    def reset(self, instruction, seed):
        self.policy.reset()

    def act(self, obs):
        return self.policy.act(obs)


def agent_class(spec: str):
    """(class, extra constructor kwargs) for an agent spec, without building an env."""
    if spec == "random":
        return RandomAgent, {}
    if spec == "oracle":
        return OracleAgent, {}
    if spec == "adaptive":
        return AdaptiveAgent, {}
    if spec == "fixed":
        return FixedAgent, {}
    if spec.startswith("vbc:"):
        if not Path(spec[4:]).exists():
            raise FileNotFoundError(spec[4:])
        return VisionBCAgent, {"checkpoint": spec[4:]}
    if spec.startswith("bc:"):
        checkpoint = spec[3:]
        if not Path(checkpoint).exists():
            raise FileNotFoundError(checkpoint)
        return BCAgent, {"checkpoint": checkpoint}
    if ":" in spec:
        module, cls = spec.split(":", 1)
        return getattr(importlib.import_module(module), cls), {}
    raise ValueError(f"unknown agent {spec!r}: use random, oracle, adaptive, fixed, bc:<checkpoint>, vbc:<checkpoint> or <module>:<Class>")


def default_name(spec: str) -> str:
    """Name under which results are grouped: the class name, or "bc-history" for a
    history-conditioned behaviour-cloning checkpoint."""
    cls, kwargs = agent_class(spec)
    if cls is VisionBCAgent:
        import torch

        ckpt = torch.load(kwargs["checkpoint"], map_location="cpu", weights_only=False)
        return "vision-bc-subgoal" if ckpt.get("use_subgoal") else "vision-bc"
    if cls is BCAgent:
        import torch

        ckpt = torch.load(kwargs["checkpoint"], map_location="cpu", weights_only=False)
        if ckpt.get("history_steps", 0):
            return "bc-history"
    return cls.name


def make_agent(spec: str, env: ComputerUseEnv) -> Agent:
    cls, kwargs = agent_class(spec)
    return cls(env, **kwargs)
