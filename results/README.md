# Baseline results, ComputerXRobot-Bench v1.0

One JSON file per agent and split (`<agent>_<split>.json`), produced by
`scripts/run_baselines.sh` at the git commit recorded in each file's `meta.git_commit`
(`meta.git_dirty` is `false`: no uncommitted changes to tracked files).

| Files | Agent |
|---|---|
| `oracle_*.json` | scripted expert, privileged |
| `random_*.json` | uniform random actions |
| `bc_s{0,1,2}_*.json` | behaviour cloning, state track, training seeds 0–2 |

Each file holds:

- `episodes`: every episode's spec (task, text, scene, seed), success, time, return and
  error counts;
- `summary`: Bench Score with stratified-bootstrap 95% CI, per task and per level;
- `meta`: benchmark version, commit, library versions, platform and worker count.

Rebuild the tables in BENCHMARK.md with:

```bash
python -m benchmark report results/*.json --table splits
python -m benchmark report results/*.json --table tasks --split test_layout
```

Evaluation is deterministic: re-running the same agent at the same commit reproduces every
episode record exactly, whatever the number of workers.
