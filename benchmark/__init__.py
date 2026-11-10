"""ComputerXRobot-Bench: a benchmark for embodied computer use.

A robot must operate a simulated computer through physical devices -- press keys on a
keyboard, drag and click a two-button mouse -- to complete desktop tasks of increasing length,
under scene shifts it was not trained on. See BENCHMARK.md for the protocol.

    python -m benchmark list                                   # tasks, splits, tracks
    python -m benchmark run --agent oracle --split test        # evaluate an agent
    python -m benchmark report results/*.json                  # tables (Markdown / LaTeX)
"""

BENCHMARK_NAME = "ComputerXRobot-Bench"
BENCHMARK_VERSION = "1.0.0"
