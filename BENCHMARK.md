# ComputerXRobot-Bench v1.0

**Embodied computer use.** A robot arm must operate a computer through its physical devices:
press keys on a 44-key keyboard and drag and click a two-button mouse. The goal is to complete
desktop tasks: typing, opening apps and files, right-click menus, web search, and multi-step
edits. Success is judged only from the state of the simulated desktop. Test episodes use
strings, queries and files held out from training (tasks with a fixed set, such as single keys
or apps, reuse it), and three test splits also shift the scene outside the training
distribution.

The benchmark measures what screen-only computer-use agents and language-conditioned
manipulation benchmarks each leave out:

- **Contact-rich device control:** millimetre key presses; a relative mouse that has to be
  servoed closed-loop on the cursor; switching between mouse buttons.
- **Grounding:** reading key legends and screen text from cameras (vision track).
- **Long-horizon composition:** up to four sub-goals across apps.
- **Robustness:** to device layout, appearance and device dynamics.

| | |
|---|---|
| Version | 1.0.0 (`benchmark/__init__.py`); results are only comparable within a major version |
| Tasks | 9 tasks in 4 difficulty levels |
| Splits | `train`, `val` (model selection), and 4 test splits of 180 episodes (20 per task): `test`, `test_layout`, `test_visual`, `test_dynamics` |
| Tracks | `state` (privileged low-dimensional state) and `vision` (cameras, proprioception, instruction text) |
| Metric | Bench Score: success averaged over tasks, with a stratified-bootstrap 95% CI |
| Bounds | scripted oracle (privileged) **100.0** on every split; random **≤ 0.6** on every split |
| Compute | CPU only; ~2,000 env steps/s per core in the state track, ~70 in the vision track |

## Contents

1. [Tasks](#tasks)
2. [Splits and generalisation axes](#splits-and-generalisation-axes)
3. [Tracks: what an agent sees](#tracks-what-an-agent-sees)
4. [Actions and control](#actions-and-control)
5. [Evaluation protocol](#evaluation-protocol)
6. [Metrics](#metrics)
7. [Baselines and results](#baselines-and-results)
8. [Reproducing the results](#reproducing-the-results)
9. [Evaluating your own agent](#evaluating-your-own-agent)
10. [Validity checks](#validity-checks)
11. [Datasheet and limitations](#datasheet-and-limitations)

## Tasks

Levels count the distinct sub-goals and skills an episode chains. Content items are the
strings, queries, apps or files a task is instantiated with; train and test items are disjoint
except where a task has only a fixed set (every key for `type_key`, every app for `open_app`).

| Level | Task | Example instruction (test) | Success when | Time limit | Items train / test |
|---|---|---|---|---|---|
| L1 | `type_key` | Press the "a" key | editor text is exactly the character | 7 s | 36 / 36 |
| L1 | `open_app` | Open the browser app | that app is in front | 12 s | 3 / 3 |
| L1 | `close_window` | Close the files window | no app is open | 12 s | 2 / 1 |
| L2 | `type_text` | Type "arm" in the text editor | editor text equals the string | 6 s + 1 s/char | 18 / 9 |
| L2 | `clear_text` | Clear the editor with the right-click menu | the prefilled editor is empty | 15 s | 5 / 3 |
| L2 | `open_file` | Open the file papers.txt | that file is open in the editor | 15 s | 3 / 2 |
| L3 | `search` | Search the web for "robotics research papers" | results page for exactly that query | 15 s + 1 s/char | 8 / 4 |
| L3 | `open_result` | Search for "…" and open "…" | that result page is open | 25 s + 1 s/char | 3 / 2 |
| L4 | `write_note` | Replace the draft in the editor with "send report" | editor shows the note (and nothing else) | 20 s + 1 s/char | 5 / 3 |

Skills involved: typing (all L2+ text tasks); left-click on dock, window, file, search and
result widgets; right-click and context-menu selection (`clear_text`, `write_note`); dragging
the mouse to a widget (every click task). A typo is recoverable: the task then expects
backspace. A missed click can be retried. There is no partial credit in the score.

## Splits and generalisation axes

| Split | Content | Scene distribution | What it tests |
|---|---|---|---|
| `train` | train items | `id` | data for imitation / RL (not a test split) |
| `val` | train items | `id` | model selection and hyperparameters (own seeds; not a test split) |
| `test` | held-out items | `id` | unseen strings and queries in familiar scenes |
| `test_layout` | held-out items | `ood_layout` | keyboard and mouse placed beyond the training range |
| `test_visual` | held-out items | `ood_visual` | unseen lighting, desk colour, floor, wallpaper, camera offset |
| `test_dynamics` | held-out items | `ood_dynamics` | unseen cursor gain (mouse DPI), key stiffness, mouse friction |

Scene distributions (`environment/variation.py`). Each OOD range is disjoint from the `id`
range on its axis and equal to `id` on the others; `tests/test_benchmark.py` checks this.

| Axis | `id` (train and `test`) | OOD |
|---|---|---|
| Keyboard offset (x, y) | ±2 cm, −2.5…+3 cm | `ood_layout`: \|x\| 3–4.5 cm, y +4…+6 cm |
| Mouse / pad offset (x, y) | ±2 cm, −3…+2 cm | `ood_layout`: \|x\| 3–4.5 cm, y −7…−4 cm |
| Light intensity, direction | ×0.8–1.2, ±0.3 rad | `ood_visual`: ×0.45–0.6 or ×1.5–1.7, \|0.6–1.0\| rad |
| Desk colour, wallpaper | 4 wood tones, 4 blues | `ood_visual`: 4 unseen desk colours, 4 unseen wallpapers |
| Floor brightness | ×0.85–1.15 | `ood_visual`: ×0.5–0.65 or ×1.4–1.6 |
| Robot-camera offset | ±0.8 cm per axis | `ood_visual`: 1.5–2.5 cm per axis |
| Cursor gain | 5,600–7,200 px/m | `ood_dynamics`: 4,000–4,800 or 8,400–9,600 px/m |
| Key stiffness | ×0.9–1.1 | `ood_dynamics`: ×1.3–1.5 |
| Mouse friction | ×0.8–1.2 | `ood_dynamics`: ×1.6–2.0 |

Episode specs (task, item, scene distribution, seed) are a pure function of the split. Seeds
are stable hashes of `(split, task, index)`, so every machine evaluates the same 180
episodes per split (`python -m benchmark list`).

## Tracks: what an agent sees

| Track | Observations | Purpose |
|---|---|---|
| `state` | proprioception (`qpos`, `qvel`, `tool_pos`, `tool_target`, `finger`); positions of the mouse, the needed mouse button and the keyboard; cursor position; key / button / palm contact flags; the instruction; and the task's **next sub-goal** (character to type, or widget to click with which button and its screen position) | isolates control; an upper-bound setting for learning |
| `vision` | four RGB cameras: `robot_cam` 240×320, `keyboard_cam` 256×512, `mouse_cam` 320×320, `screen_cam` 400×640; proprioception; palm touch; the instruction text. **No** positions, cursor or sub-goals | the full problem: read keys and screen, find the cursor, plan |

The harness removes everything a track does not allow before observations reach the agent.
The scripted oracle reads the simulator directly. It is reported as a privileged reference,
not as an entry in either track.

## Actions and control

A 50 Hz action `[dx, dy, dz, finger]`:

- `dx, dy, dz` move the hand's target; each component is clipped to ±6 mm per step.
- `finger` is the extension of a 30 mm finger, from 0 to 1.

Damped-least-squares IK and joint position servos (500 Hz physics) track the target with the
hand pointing down. Key and button presses register when a key or button travels past 60% of
its stroke. The mouse is relative: cursor motion is mouse travel × cursor gain. The Gymnasium
wrapper (`environment/gym_env.py`) exposes the same action normalized to [−1, 1].

## Evaluation protocol

1. **Train only on the `train` split** (content items and `id` scenes). Generating more data
   from that distribution (e.g. `ComputerXRobot/*-v0` with `split="train"`, `scene="id"`) is
   allowed. Select models and hyperparameters on `val`. Test splits must not be used for
   training, tuning or model selection.
2. **Evaluate on all four test splits** with the official episode lists: 20 per task, 180 per
   split, fixed seeds. Do not cherry-pick tasks or episodes.
3. **Report the track.** Results are only comparable within a track. Agents that read
   simulator state outside their track are marked privileged.
4. **Learned agents: at least 3 training seeds,** reported as mean ± std of the Bench Score
   over seeds. A single run is reported with its bootstrap 95% CI.
5. **Report per-task and per-level success** and the error metrics alongside the score
   (`python -m benchmark report --table tasks|levels|errors`).

All agents see identical episodes, so paired per-episode tests (e.g. McNemar) can compare two
agents on the same split.

## Metrics

| Metric | Definition |
|---|---|
| **Bench Score** | success rate per task, averaged over the 9 tasks (each task weighs the same) |
| 95% CI | stratified bootstrap: episodes resampled within each task, 2,000 resamples |
| Per-level success | mean success of the tasks of each level (L1–L4) |
| Time to success | simulated seconds, successful episodes only |
| Wrong keys | key presses that were not the character the task needed at that moment |
| Failed clicks | clicks that hit nothing or the wrong widget, or used the wrong button |
| Collisions | robot contacts with anything except key tops and mouse surfaces (onsets) |
| Failed contacts | the finger touched a key or button without actuating it |

## Baselines and results

Three baselines on the official splits. The scripted oracle is reported as a privileged
reference; it reads the simulator state directly.

* **random**: uniform random actions (hand moves within ±6 mm per step, finger 0–1).
* **oracle (privileged)**: the scripted expert (`learning/scripted_policy.py`). It establishes
  that every episode is solvable.
* **bc** (state track): an MLP behaviour-cloning policy (`learning/policy.py`).
  * Inputs: the 70 state features.
  * Outputs: the expert's waypoint offset and finger command.
  * Data: 100 oracle demonstrations per task on `train` (893 kept, 179k samples). DART-style
    perturbations of the executed motion, 30% strength during key presses and mouse drags.
  * Model: 3×256 MLP, smooth-L1 loss (β = 0.05), AdamW (lr 10⁻³, cosine), batch 512, 80 epochs.
  * Seeds: 3, reported as mean ± std.

![BC Bench Score per split and per level](docs/benchmark_results.png)

*Left: BC's Bench Score per test split (mean of 3 seeds, error bars = std). Right: its
success per difficulty level.*

### Bench Score per split

Mean ± std over 3 seeds for `bc`; bootstrap 95% CI half-width for single runs.

| Agent | Track | test | test_layout | test_visual | test_dynamics | val |
|---|---|---|---|---|---|---|
| bc | state | 96.1 ± 4.0 | 63.9 ± 9.9 | 94.8 ± 5.8 | 84.8 ± 1.5 | 96.1 ± 5.1 |
| random | state | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.6 ± 0.8 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| oracle (privileged) | state | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 |

### Success % per task: `test`

| Agent | Track | type_key (L1) | open_app (L1) | close_window (L1) | type_text (L2) | clear_text (L2) | open_file (L2) | search (L3) | open_result (L3) | write_note (L4) | Score |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bc | state | 100 | 100 | 100 | 100 | 90 | 88 | 100 | 100 | 87 | 96.1 ± 4.0 |
| random | state | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 ± 0.0 |
| oracle (privileged) | state | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100.0 ± 0.0 |

### Success % per task: `test_layout`

| Agent | Track | type_key (L1) | open_app (L1) | close_window (L1) | type_text (L2) | clear_text (L2) | open_file (L2) | search (L3) | open_result (L3) | write_note (L4) | Score |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bc | state | 100 | 63 | 65 | 95 | 20 | 55 | 67 | 77 | 33 | 63.9 ± 9.9 |
| random | state | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 ± 0.0 |
| oracle (privileged) | state | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100.0 ± 0.0 |

### Success % per task: `test_visual`

| Agent | Track | type_key (L1) | open_app (L1) | close_window (L1) | type_text (L2) | clear_text (L2) | open_file (L2) | search (L3) | open_result (L3) | write_note (L4) | Score |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bc | state | 100 | 100 | 100 | 100 | 82 | 85 | 100 | 100 | 87 | 94.8 ± 5.8 |
| random | state | 0 | 0 | 0 | 0 | 5 | 0 | 0 | 0 | 0 | 0.6 ± 0.8 |
| oracle (privileged) | state | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100.0 ± 0.0 |

### Success % per task: `test_dynamics`

| Agent | Track | type_key (L1) | open_app (L1) | close_window (L1) | type_text (L2) | clear_text (L2) | open_file (L2) | search (L3) | open_result (L3) | write_note (L4) | Score |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bc | state | 100 | 72 | 85 | 97 | 85 | 75 | 100 | 100 | 50 | 84.8 ± 1.5 |
| random | state | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 ± 0.0 |
| oracle (privileged) | state | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100.0 ± 0.0 |

### Success % per level

| Split | L1 | L2 | L3 | L4 |
|---|---|---|---|---|
| `test` (bc) | 100 | 93 | 100 | 87 |
| `test_layout` (bc) | 76 | 57 | 72 | 33 |
| `test_visual` (bc) | 100 | 89 | 100 | 87 |
| `test_dynamics` (bc) | 86 | 86 | 100 | 50 |

### Errors and speed on `test`

| Agent | Track | Wrong keys | Failed clicks | Collisions | Failed contacts | Time to success (s) |
|---|---|---|---|---|---|---|
| bc | state | 0.00 | 0.00 | 0.00 | 0.01 | 4.2 |
| random | state | 2.41 | 0.04 | 17.55 | 0.99 | – |
| oracle (privileged) | state | 0.00 | 0.00 | 0.00 | 0.10 | 4.2 |

### Model selection for `bc`

Recipes were compared on `val` (10 episodes per task, seed 0). The selected one is in bold.

| Recipe (100 demonstrations per task) | `val` score |
|---|---|
| 3×256 MLP, 40 epochs | 76.7 |
| 3×512 MLP, 60 epochs, near-L1 loss (β = 0.005) | 23.3 |
| 3×512 MLP, 40 epochs | 98.9 |
| **3×256 MLP, 80 epochs** | **100.0** |

Disclosure: before `val` existed, a first recipe (40 demonstrations per task, almost no
perturbation during mouse drags, 40 epochs) was evaluated once on `test` and scored 22.2. Its
diagnosis led to the two changes above: more data and perturbations during drags. After that,
all selection was done on `val`.

### Findings

1. **Unseen content in familiar scenes is close to solved in the state track:** 96.1 ± 4.0.
   Given the next sub-goal, an MLP generalises to strings and queries it never saw, and on
   `test` it is as fast as the oracle (4.2 s mean time to success) with almost no errors.
2. **Device layout is the hardest shift:** 63.9 ± 9.9, a 32-point drop.
   - BC still types: `type_key` 100, `type_text` 95.
   - It loses the mouse when the mouse is placed beyond its training range: `clear_text` 20,
     `write_note` 33, `open_app` 63.
3. **Device dynamics cost 11 points:** 84.8 ± 1.5. An unseen cursor gain and mouse friction
   mainly hurt the right-click and long-horizon tasks (`write_note` 50).
4. **Visual shift does not affect the state track** (94.8), as designed. It targets the
   vision track.
5. **Longer tasks degrade most under shift.** L4 drops to 33% on `test_layout` and 50% on
   `test_dynamics`. Seed variance under shift is large (±9.9 on `test_layout`), so learned
   agents need several seeds.
6. **The vision track is open.** No learned agent has been evaluated on it yet; it requires
   reading keys and screen and planning without sub-goals.

## Reproducing the results

```bash
pip install -r requirements.txt
sh scripts/run_baselines.sh          # PY=python WORKERS=8; ~15 min on a 10-core laptop
python -m benchmark report results/*.json --table splits            # Markdown tables
python -m benchmark report results/*.json --table tasks --split test --format latex
python -m benchmark.plot results/*.json --agent bc --out docs/benchmark_results.png
```

`scripts/run_baselines.sh` does the following:

1. Collects oracle demonstrations on `train`: 100 per task, `id` scenes, DART-style
   perturbations (~3 min).
2. Trains three behaviour-cloning seeds (~30 s each on CPU).
3. Evaluates the oracle, random and the three seeds on `val` and every test split (~9 min).

Each result file (`results/<agent>_<split>.json`) contains:

- every episode record (spec, success, time, return, error counts);
- the summary;
- metadata: benchmark version, git commit, whether the tree had uncommitted changes,
  MuJoCo / NumPy / pygame versions, platform, worker count.

Results do not depend on the number of workers (`tests/test_benchmark.py`).

## Evaluating your own agent

```python
# my_agent.py
import numpy as np
from benchmark.agents import Agent

class MyAgent(Agent):
    name = "my-agent"
    track = "vision"                      # or "state"

    def reset(self, instruction: str, seed: int) -> None:
        ...

    def act(self, obs: dict) -> np.ndarray:  # [dx, dy, dz] metres (|.| <= 0.006), finger 0..1
        ...
```

```bash
python -m benchmark run --agent my_agent:MyAgent --all-test --workers 8 --tag my-agent
python -m benchmark report results/*.json
```

From Python, `benchmark.harness.evaluate("my_agent:MyAgent", "test", workers=8, out=...)` does
the same. With `workers > 1` it uses spawned processes, so call it under
`if __name__ == "__main__":`.

`benchmark/example_agent.py` lists every observation of the vision track with its shape.
`env.annotations(camera)` provides ground-truth boxes and labels for keys, mouse buttons and
screen text in any camera image, for training perception modules on `train` scenes (see
README).

## Validity checks

`python -m pytest -q` runs:

- **Solvable:** the oracle succeeds on every task in every test split, with no wrong keys,
  failed clicks or collisions. On the full splits it scores 100.0.
- **Not trivial:** the random agent scores at most 0.6 on any split (full results).
- **Fixed and reproducible:** episode lists are identical across calls and unique per split,
  and parallel and sequential evaluation give identical records.
- **Genuinely out of distribution:** the OOD ranges are disjoint from `id` on their axis
  (sampled check).
- **Tracks are isolated:** the vision track never receives positions, cursor, sub-goals or
  the screen framebuffer.
- **Legible:** all 44 key legends, the L/R mouse buttons and the screen text are read back
  from rendered camera images by template matching (`tests/test_legibility.py`).

## Datasheet and limitations

- **Simulator:** MuJoCo 3 with a custom 5-DOF arm and finger; pygame desktop rendered onto
  the monitor texture. No real-robot or sim-to-real claims.
- **Desktop:** a hand-built GUI (editor, file manager, offline browser with 12 pages, context
  menus), not a real OS. Lowercase keyboard without modifier keys; the scroll wheel is
  visual only.
- **Demonstrations:** `python -m benchmark collect` produces oracle trajectories with features,
  waypoint labels and optionally 64×64 images (`learning/dataset.py`). They contain no human
  data or personal information.
- **The `state` track hands over the next sub-goal.** It measures control, not planning or
  perception. The `vision` track has no learned baseline yet and is the open challenge.
- **Content set sizes:** `close_window` has one test item and `open_result` two. Their test
  episodes differ mainly in scene, which the per-task tables make visible.
- **Versioning:** any change to tasks, splits, scene ranges or success checks bumps the major
  version.

## Citation and licence

Code, assets and benchmark definitions are released under the MIT licence (`LICENSE`).

```bibtex
@misc{computerxrobot_bench_2026,
  title        = {ComputerXRobot-Bench: Embodied Computer Use with a Physical Keyboard and Mouse},
  author       = {Dhiman, Anurag},
  year         = {2026},
  note         = {Version 1.0.0},
  howpublished = {\url{https://github.com/Anurag9Dhiman/ComputerXRobot}}
}
```
