"""Offline "web" for the simulated browser: a small fictional corpus and a keyword search."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class WebPage:
    id: str
    title: str
    url: str
    snippet: str
    body: str
    keywords: tuple[str, ...] = ()


PAGES: tuple[WebPage, ...] = (
    WebPage(
        "robot-typing",
        "Learning to Type with a Robot Arm",
        "papers.example/robot-typing",
        "We train a robot arm to press physical keys from camera images and joint states.",
        "Typing requires millimetre accuracy and well-timed contact. We collect scripted "
        "demonstrations, train a behaviour cloning policy and measure character accuracy on "
        "held-out sentences.",
        ("robotics", "research", "papers", "keyboard", "typing", "imitation"),
    ),
    WebPage(
        "sim-computer-use",
        "Robots Using Computers in Simulation",
        "papers.example/sim-computer-use",
        "A survey of embodied agents that operate keyboards, mice and touch screens.",
        "Embodied computer use couples manipulation with interface understanding. Simulated "
        "desktops make it possible to generate unlimited tasks with exact success checks.",
        ("robotics", "research", "papers", "survey", "computer", "simulation", "mouse"),
    ),
    WebPage(
        "diffusion-policy",
        "Diffusion Policies for Visuomotor Control",
        "papers.example/diffusion-policy",
        "Action diffusion models capture multimodal demonstrations for manipulation.",
        "Instead of regressing a single action, the policy denoises a short action sequence "
        "conditioned on recent observations.",
        ("robotics", "research", "papers", "diffusion", "policy", "learning"),
    ),
    WebPage(
        "mujoco-intro",
        "Getting Started with MuJoCo",
        "docs.example/mujoco",
        "Physics simulation for robotics: models, contacts, actuators and rendering.",
        "MuJoCo models are written in MJCF. Bodies carry geoms, joints and sites; actuators "
        "apply forces; the renderer can draw any camera offscreen.",
        ("mujoco", "simulation", "physics", "robotics", "tutorial"),
    ),
    WebPage(
        "rl-basics",
        "Reinforcement Learning: PPO and SAC Explained",
        "learn.example/rl-basics",
        "On-policy and off-policy methods compared with simple examples.",
        "PPO optimises a clipped surrogate objective; SAC maximises reward plus entropy with "
        "a replay buffer. Both are common baselines for continuous control.",
        ("reinforcement", "learning", "ppo", "sac", "rl", "tutorial"),
    ),
    WebPage(
        "behavior-cloning",
        "A Practical Guide to Behaviour Cloning",
        "learn.example/behavior-cloning",
        "Supervised learning from demonstrations, and how to fight compounding errors.",
        "Behaviour cloning fits a policy to expert actions. Noise injection while collecting "
        "data (DART) teaches the policy to recover from its own mistakes.",
        ("imitation", "learning", "behavior", "cloning", "demonstrations", "tutorial"),
    ),
    WebPage(
        "vision-transformers",
        "Vision Transformers for Robot Perception",
        "papers.example/vit-robots",
        "Patch-based encoders as visual backbones for manipulation policies.",
        "Pretrained vision transformers provide strong features for grasping and pointing "
        "tasks, especially with limited robot data.",
        ("vision", "transformer", "robotics", "research", "papers", "perception"),
    ),
    WebPage(
        "pasta-recipe",
        "Quick Tomato Pasta Recipe",
        "food.example/tomato-pasta",
        "A twenty minute dinner with tomatoes, garlic and basil.",
        "Boil the pasta, fry garlic in olive oil, add tomatoes and basil, then toss together.",
        ("pasta", "recipe", "cooking", "dinner", "food"),
    ),
    WebPage(
        "weather-today",
        "Weather Forecast for Today",
        "weather.example/today",
        "Sunny in the morning, light rain expected in the evening.",
        "Highs of 24 degrees with a light breeze. Take an umbrella if you are out late.",
        ("weather", "forecast", "rain", "today"),
    ),
    WebPage(
        "python-lists",
        "Python Lists and Dictionaries",
        "learn.example/python-lists",
        "The two workhorse data structures of Python, with examples.",
        "Lists keep items in order; dictionaries map keys to values. Both are mutable.",
        ("python", "programming", "tutorial", "lists"),
    ),
    WebPage(
        "mouse-control",
        "Fine Mouse Control with Robot Hands",
        "papers.example/mouse-control",
        "Dragging and clicking a physical mouse with closed-loop cursor feedback.",
        "The robot watches the cursor on screen and corrects its hand motion until the "
        "pointer reaches the target, then presses the button.",
        ("robotics", "research", "papers", "mouse", "cursor", "control"),
    ),
    WebPage(
        "travel-tokyo",
        "Three Days in Tokyo",
        "travel.example/tokyo",
        "An itinerary covering markets, museums and parks.",
        "Day one: markets and temples. Day two: museums. Day three: parks and gardens.",
        ("travel", "tokyo", "itinerary", "japan"),
    ),
)

PAGE_BY_ID = {page.id: page for page in PAGES}

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def search(query: str, limit: int = 5) -> list[WebPage]:
    """Rank pages by how many query words appear in their title, snippet or keywords."""
    terms = set(tokenize(query))
    if not terms:
        return []
    scored = []
    for order, page in enumerate(PAGES):
        words = set(tokenize(page.title + " " + page.snippet)) | set(page.keywords)
        score = len(terms & words)
        if score:
            scored.append((-score, order, page))
    return [page for _, _, page in sorted(scored)[:limit]]
