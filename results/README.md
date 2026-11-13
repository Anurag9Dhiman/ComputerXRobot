# Baseline results, ComputerXRobot-Bench v1.1

One JSON file per agent and split (`<agent>_<split>.json`), produced at the git commit
recorded in each file's `meta.git_commit`. `meta.git_dirty` is `false` in every file: there were
no uncommitted changes to tracked files.

| Files | Agent | Script |
|---|---|---|
| `oracle_*.json` | scripted expert, privileged | `run_baselines.sh`, `run_hard_baselines.sh` |
| `random_*.json` | uniform random actions | `run_baselines.sh`, `run_hard_baselines.sh` |
| `bc_s{0,1,2}_*.json` | behaviour cloning on `train`, state track, training seeds 0–2 | `run_baselines.sh` |
| `nominal_bc_s{0,1,2}_hard_*.json` | the same checkpoints, zero-shot on the hard tier (`bc-nominal`) | `run_hard_baselines.sh` |
| `hard_bc_s{0,1,2}_*.json` | behaviour cloning on `hard_train` (`bc`) | `run_hard_baselines.sh` |
| `hard_bch_s{0,1,2}_*.json` | the same with a 4-step history (`bc-history`) | `run_hard_baselines.sh` |
| `hard_bcd_s{0,1,2}_*.json` | `bc` after two DAgger iterations (`bc-dagger`) | `run_hard_dagger.sh` |
| `hard_bchd_s{0,1,2}_*.json` | `bc-history` after two DAgger iterations (`bc-history-dagger`) | `run_hard_dagger.sh` |
| `adaptive_*.json`, `fixed_*.json` | non-privileged scripted experts, with and without online identification of the hidden dynamics | `run_adaptive.sh` |
| `ablation_*.json` | dynamics ablation on `hard_test`, one file per agent and seed | `run_hard_ablation.sh`, `run_adaptive.sh` |
| `world_model.json` | dynamics-prediction track: persistence, kinematic and MLP (3 seeds) | `benchmark.world_model` |

Each episode-result file holds:

- `episodes`: every episode's spec (task, text, scene, seed and, in the hard tier, the
  dynamics distribution), success, time, return and error counts;
- `summary`: Bench Score with stratified-bootstrap 95% CI, per task and per level;
- `meta`: benchmark version, commit, library versions, platform and worker count.

Rebuild the tables in BENCHMARK.md with:

```bash
python -m benchmark report results/*.json --table splits
python -m benchmark report results/*.json --table hard
python -m benchmark report results/*.json --table tasks --split hard_test_ood
```

Evaluation is deterministic: re-running the same agent at the same commit reproduces every
episode record exactly, whatever the number of workers. A hard-tier episode's hidden dynamics
follow from its seed (`EpisodeSpec.hidden_dynamics()`).
