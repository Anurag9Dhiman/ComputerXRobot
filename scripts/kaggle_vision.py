# ComputerXRobot — vision-track baseline on Kaggle
# ============================================================
# Paste each block below into its own Kaggle notebook cell.
# Settings: Accelerator = GPU T4 x2 (or P100), Internet = ON.
# The pipeline is RESUMABLE: every step skips outputs that already exist,
# so if a 9-hour session ends mid-run, commit the notebook, add its output
# as an input dataset next session, copy it back into /kaggle/working, and
# re-run — it continues where it stopped.
#
# CHECK CELL 1'S PROBE FIRST: it times one rendered episode. GPU/EGL ~ a few seconds;
# if it reports 30s+, rendering is on the CPU and the run won't fit a session — fix EGL first.
# First-pass settings that fit one GPU session (set before running cell 2/3):
#   os.environ["PER_TASK"]="40"; os.environ["SEEDS"]="0"; os.environ["EPOCHS"]="30"

# %% [cell 1] Setup: clone (public repo), install, headless rendering
import os, subprocess, sys, pathlib

WORK = pathlib.Path("/kaggle/working")
REPO = WORK / "ComputerXRobot"
if not REPO.exists():
    subprocess.run(["git", "clone", "--depth", "1", "-b", "main",
                    "https://github.com/Anurag9Dhiman/ComputerXRobot.git", str(REPO)], check=True)
os.chdir(REPO)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "."], check=True)

# Headless GPU rendering (matches what evaluation sees). If EGL fails, fall back to CPU osmesa.
os.environ["MUJOCO_GL"] = "egl"
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import time, torch
assert torch.cuda.is_available(), "No GPU! Set Settings -> Accelerator -> GPU T4 x2, then re-run."
print("CUDA:", torch.cuda.is_available(), "|", torch.cuda.get_device_name(0))

# Which MuJoCo render backend actually engaged? egl = GPU (fast); osmesa/glfw = CPU (10-50x slower).
import mujoco
print("MUJOCO_GL =", os.environ.get("MUJOCO_GL"))

# RENDER-SPEED PROBE: collect 1 vision episode and time it. This is the number that decides feasibility.
# On a GPU with EGL this should be a few seconds. If it takes ~30s+, rendering is on the CPU (osmesa) and
# the full run will NOT fit a session -- fix EGL before collecting (do not proceed).
t0 = time.time()
r = subprocess.run([sys.executable, "-m", "benchmark", "collect", "--split", "train", "--tasks", "type_key",
                    "--per-task", "1", "--vision", "--seed", "999", "--out", "/kaggle/working/probe.npz"],
                   capture_output=True, text=True)
dt = time.time() - t0
print(r.stdout[-400:], r.stderr[-600:])
print(f"\n>>> PROBE: 1 vision episode took {dt:.1f}s")
print(">>> GPU/EGL expected: a few seconds. If 30s+, it's CPU rendering -- STOP and fix EGL before cell 2.")

# %% [cell 2] Collect oracle demonstrations (vision inputs). Resumable: skips existing .npz.
import os, subprocess, sys
TASKS = "type_key open_app close_window type_text clear_text open_file search open_result write_note".split()
PER_TASK = int(os.environ.get("PER_TASK", 100))
os.makedirs("data", exist_ok=True)
procs = []
for i, t in enumerate(TASKS):
    out = f"data/vision_train_{t}.npz"
    if os.path.exists(out):
        print("skip", out); continue
    log = open(f"data/vision_train_{t}.log", "w")
    procs.append(subprocess.Popen([sys.executable, "-m", "benchmark", "collect", "--split", "train",
        "--tasks", t, "--per-task", str(PER_TASK), "--seed", str(200 + i), "--vision", "--out", out],
        stdout=log, stderr=subprocess.STDOUT))
    # Kaggle has few CPU cores: cap concurrent collectors so rendering doesn't thrash.
    if len(procs) >= max(1, (os.cpu_count() or 4) - 1):
        for p in procs: p.wait()
        procs = []
for p in procs: p.wait()
DATA = [f"data/vision_train_{t}.npz" for t in TASKS]
print("collected:", [f for f in DATA if os.path.exists(f)])

# %% [cell 3] Train vbc + vbc-subgoal per seed on the GPU. Resumable: skips existing .pt.
import os, subprocess, sys
SEEDS = os.environ.get("SEEDS", "0 1 2").split()
EPOCHS = int(os.environ.get("EPOCHS", 30))
os.makedirs("checkpoints", exist_ok=True)
for s in SEEDS:
    for subgoal, name in [([], "vbc"), (["--subgoal"], "vbcsg")]:
        ckpt = f"checkpoints/{name}_s{s}.pt"
        if os.path.exists(ckpt):
            print("skip", ckpt); continue
        print("=== training", ckpt, "===", flush=True)
        subprocess.run([sys.executable, "-m", "learning.vision", "train", "--data", *DATA,
            "--epochs", str(EPOCHS), "--seed", s, "--device", "cuda", "--out", ckpt] + subgoal, check=True)

# %% [cell 4] Evaluate (CPU rendering). vbc on 4 test splits, vbc-subgoal on test.
import os, subprocess, sys
W = str(max(1, (os.cpu_count() or 4) - 1))
os.makedirs("results", exist_ok=True)
for s in SEEDS:
    if not os.path.exists(f"results/vbc_s{s}_test_dynamics.json"):
        subprocess.run([sys.executable, "-m", "benchmark", "run", "--agent", f"vbc:checkpoints/vbc_s{s}.pt",
            "--tag", f"vbc_s{s}", "--all-test", "--workers", W, "--out-dir", "results"], check=True)
    if not os.path.exists(f"results/vbcsg_s{s}_test.json"):
        subprocess.run([sys.executable, "-m", "benchmark", "run", "--agent", f"vbc:checkpoints/vbcsg_s{s}.pt",
            "--tag", f"vbcsg_s{s}", "--track", "vision_subgoal", "--split", "test", "--workers", W,
            "--out-dir", "results"], check=True)

# %% [cell 5] Report + plot. These numbers go into the paper's vision table.
import subprocess, sys, glob
print(subprocess.run([sys.executable, "-m", "benchmark", "report", *glob.glob("results/*.json"),
                      "--table", "splits"], capture_output=True, text=True).stdout)
print(subprocess.run([sys.executable, "-m", "benchmark", "report", *glob.glob("results/vbcsg_*.json"),
                      "--table", "aggregate", "--split", "test"], capture_output=True, text=True).stdout)
subprocess.run([sys.executable, "-m", "benchmark.plot", *glob.glob("results/*.json"), "--vision",
                "--out", "/kaggle/working/vision_results.png"])
# Download results/ and vision_results.png from the Kaggle output, commit them to the repo:
#   git add results/vbc_s*_test*.json results/vbcsg_s*_test.json docs/vision_results.png
