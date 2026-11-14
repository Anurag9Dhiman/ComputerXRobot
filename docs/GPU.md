# Runs on a GPU server

Every state-track result (`scripts/run_cpu.sh`, 5 seeds) runs on a laptop CPU, in about a
day with the recurrent baseline and the dynamics ablation. Those results are in `results/`.
The one remaining result that needs a GPU is the **vision-track baseline**: behaviour cloning from
pixels (`vision-bc`, `learning/vision.py`) and its diagnostic variant with the next sub-goal
(`vision-bc-subgoal`), three seeds each, 30 epochs over ~89k samples. On an Apple-silicon
laptop GPU this takes ~2 h per model, ~12 h in all. On an A100 it is expected to take minutes
per model.

| Step | Where it runs | Laptop (M-series) | A100 server |
|---|---|---|---|
| Collect 100 oracle demonstrations per task with the 4 cameras | CPU, one process per task | ~31 min | similar (9 processes) |
| Train 6 models × 30 epochs | GPU | ~12 h (MPS) | expected well under 1 h per model |
| Evaluate: `vision-bc` on 4 test splits, `vision-bc-subgoal` on `test` | CPU, `WORKERS` processes | ~4 h with 6 workers | ~30 min with 32 workers |

## 1. Set up

```bash
git clone https://github.com/Anurag9Dhiman/ComputerXRobot.git && cd ComputerXRobot
git checkout feature/publication-baselines         # until its PR (the one adding this file) is merged; then main
```

Either conda:

```bash
conda env create -f environment.yml && conda activate robot-sim
pip install -e .                                   # adds the cxr-bench command
nvidia-smi && python -c "import torch; print(torch.cuda.is_available())"
```

or Docker (`--gpus all` needs the NVIDIA Container Toolkit):

```bash
docker build -t computerxrobot .
docker run --gpus all --rm -it -v "$PWD/data:/app/data" -v "$PWD/checkpoints:/app/checkpoints" \
    -v "$PWD/results:/app/results" -v "$PWD/docs:/app/docs" -e WORKERS=32 computerxrobot sh scripts/run_gpu.sh
```

On a Linux machine without a display the package selects headless rendering itself
(`MUJOCO_GL=egl`, `SDL_VIDEODRIVER=dummy`; `environment/__init__.py`). If EGL is unavailable
(no NVIDIA driver with graphics capability), `export MUJOCO_GL=osmesa` renders on the CPU
instead (`apt install libosmesa6`), at a lower speed.

## 2. Run

```bash
WORKERS=$(nproc) nohup sh scripts/run_gpu.sh > outputs/gpu_run.log 2>&1 &
tail -f outputs/gpu_run.log
```

`scripts/run_gpu.sh` checks the GPU, renders one vision-track episode as a smoke test and then
runs `scripts/run_vision.sh`. Every step whose output exists is skipped, so after an
interruption the same command resumes. Set `EPOCHS` to change the budget (default 30).

Collect the demonstrations on the server rather than copying them from another machine:
training and evaluation images should come from the same renderer (EGL here, CGL on a Mac).

## 3. Bring the results back

```bash
git add results/vbc_s*_test*.json results/vbcsg_s*_test.json docs/vision_results.png
git commit -m "Vision-track baseline results"
git push
```

The result files carry the commit they were produced with (`meta.git_commit`, and
`meta.git_dirty` is false only if no tracked code was modified). Then update BENCHMARK.md and
README.md with:

```bash
python -m benchmark report results/*.json --table splits      # vision-bc row
python -m benchmark report results/vbcsg_*.json --table aggregate --split test
python -m benchmark report results/*.json --table tasks --split test
```

## 4. Optional, if GPU time is left

- **A bigger vision budget**: more epochs (`EPOCHS=60`) or more demonstrations
  (`PER_TASK=200`; delete `data/vision_train_*.npz` first). At 8 epochs the policy is
  under-trained (3–9 mm position error near the goal). Whether 30 epochs suffices for typing
  precision is the open question this run answers.
- **More vision seeds**: `SEEDS="0 1 2 3 4" sh scripts/run_gpu.sh` adds seeds 3 and 4.
