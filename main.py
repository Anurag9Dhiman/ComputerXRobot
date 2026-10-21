"""ComputerXRobot: a MuJoCo robot arm that uses a simulated computer with a physical
keyboard and mouse.

    python main.py                         # milestone 1: press "a", verify it in the editor
    python main.py demo --task search --text "robotics research papers" --display
    python main.py demo --task search --video outputs/search.mp4
    python main.py demo --task search --gif docs/demo.gif --speed 2
    python main.py viewer --task search    # interactive MuJoCo viewer (mjpython on macOS)
    python main.py annotate --camera keyboard_cam --out outputs/keyboard_cam.png
    python main.py collect --episodes 30 --out data/demos.npz
    python main.py train --data data/demos.npz --out checkpoints/bc.pt
    python main.py evaluate --policy bc --checkpoint checkpoints/bc.pt --split both

Tasks: type_key, type_text, search, open_result, open_app (see environment/task_manager.py).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import environment  # noqa: F401  (sets MUJOCO_GL / pygame env vars before anything else)
from environment.computer_env import ComputerUseEnv, EnvConfig
from environment.task_manager import TASKS, make_task

ALL_TASKS = list(TASKS)


# --------------------------------------------------------------------------- helpers
def env_config_for(args) -> EnvConfig:
    """Observation settings: images only when an image-based BC policy needs them."""
    checkpoint = getattr(args, "checkpoint", None)
    if getattr(args, "policy", "scripted") == "bc" and checkpoint:
        import torch

        shapes = torch.load(checkpoint, map_location="cpu", weights_only=False).get("image_shapes", {})
        if shapes:
            cfg = EnvConfig(render_obs=True)
            if "robot_cam" in shapes:
                cfg.obs_image_size = tuple(shapes["robot_cam"][:2])
            if "screen" in shapes:
                cfg.obs_screen_size = tuple(shapes["screen"][:2])
            return cfg
    if getattr(args, "images", False):
        return EnvConfig(render_obs=True, obs_image_size=(64, 64), obs_screen_size=(64, 96))
    return EnvConfig(render_obs=False)


def make_policy(args, env: ComputerUseEnv):
    if args.policy == "scripted":
        from learning.scripted_policy import ScriptedPolicy

        return ScriptedPolicy(env)
    if not args.checkpoint:
        sys.exit("--policy bc needs --checkpoint")
    from learning.policy import BCPolicy

    return BCPolicy(args.checkpoint, env.config.max_step, device=getattr(args, "device", "auto"))


def add_policy_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--policy", choices=("scripted", "bc"), default="scripted")
    p.add_argument("--checkpoint", help="BC checkpoint (.pt) for --policy bc")
    p.add_argument("--device", default="auto", help="torch device for --policy bc")


# ------------------------------------------------------------------------------ demo
def cmd_demo(args) -> int:
    from evaluation.task_success import run_episode

    env = ComputerUseEnv(env_config_for(args))
    task = make_task(args.task, args.text)
    policy = make_policy(args, env)
    print(f"Task: {task.instruction}   (policy: {args.policy})")

    def steps_per_frame(fps: float) -> int:
        """Control steps between captured frames so playback runs at `args.speed`."""
        return max(1, round(args.speed / (fps * env.config.control_dt)))

    dashboard = window = clock = None
    recorders: list[tuple[object, int]] = []  # (writer, capture every n control steps)
    if args.display or args.video or args.gif or args.screenshot:
        import pygame

        from evaluation.dashboard import HEIGHT, WIDTH, Dashboard

        dashboard = Dashboard(env, main_camera=args.camera)
        if args.display:
            pygame.display.init()
            window = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("ComputerXRobot")
            clock = pygame.time.Clock()
        if args.video:
            recorders.append((_VideoWriter(args.video, args.fps), steps_per_frame(args.fps)))
        if args.gif:
            recorders.append((_GifWriter(args.gif, args.gif_fps, args.gif_width), steps_per_frame(args.gif_fps)))
    display_every = steps_per_frame(args.fps)

    def show(frame_surface):
        import pygame

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                raise KeyboardInterrupt
        window.blit(frame_surface, (0, 0))
        pygame.display.flip()
        clock.tick(args.fps)

    def on_step(env_, obs, info):
        for ev in info.events:
            mark = "ok" if ev.correct else "UNEXPECTED"
            shown = repr(ev.value) if ev.kind == "key" else ev.value
            print(f"  t={ev.time:6.2f}s  {ev.kind:5s} {shown}  [{mark}]")
        if dashboard is None:
            return
        step = env_.step_count
        due = [writer for writer, every in recorders if step % every == 0]
        display_due = window is not None and step % display_every == 0
        if due or display_due:
            frame = dashboard.draw(getattr(policy, "skill", None), args.policy)
            if display_due:
                show(frame)
            for writer in due:
                writer.append(frame)

    t0 = time.time()
    try:
        metrics = run_episode(env, policy, task, on_step)
    except KeyboardInterrupt:
        print("interrupted")
        return 130
    wall = time.time() - t0

    if dashboard is not None:
        final = dashboard.draw(getattr(policy, "skill", None), args.policy)
        if args.screenshot:
            import pygame

            Path(args.screenshot).parent.mkdir(parents=True, exist_ok=True)
            pygame.image.save(final, args.screenshot)
            print(f"saved screenshot {args.screenshot}")
        for writer, _ in recorders:
            writer.finish(final)  # ends on the result, held on screen for a moment
            print(f"saved {writer.path}")

    _print_result(env, metrics, wall)

    if window is not None:
        import pygame

        print("Close the window (or press Esc) to exit.")
        while True:
            event = pygame.event.wait()
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                break
    env.close()
    return 0 if metrics["success"] else 1


def _print_result(env, metrics, wall) -> None:
    state = env.desktop.state()
    print()
    print(f"Result: {'SUCCESS' if metrics['success'] else 'FAILED'}  "
          f"(sim time {metrics['time']:.2f}s, wall {wall:.2f}s, {metrics['steps']} control steps)")
    if env.task.name in ("type_key", "type_text"):
        print(f"  editor text: {state['editor_text']!r}   (target {env.task.text!r})")
    elif env.task.name == "clear_text":
        print(f"  editor text: {state['editor_text']!r}   (started as {env.task.text!r})")
    elif env.task.name in ("search", "open_result"):
        print(f"  browser: page={state['browser_page']} query={state['browser_last_query']!r} "
              f"doc={state['browser_doc']}")
    else:
        print(f"  active app: {state['active_app']}")
    print(f"  key presses {metrics['key_presses']}, clicks {metrics['clicks']}, wrong keys {metrics['wrong_keys']}, "
          f"failed clicks {metrics['failed_clicks']}, collisions {metrics['collisions']}, "
          f"failed contacts {metrics['failed_contacts']}")


class _VideoWriter:
    """MP4 via imageio-ffmpeg when available, otherwise a folder of PNG frames."""

    def __init__(self, path: str, fps: int):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.count = 0
        try:
            import imageio.v2 as imageio

            self.writer = imageio.get_writer(self.path, fps=fps, codec="libx264", quality=7, macro_block_size=8)
        except Exception as exc:  # noqa: BLE001 - imageio/ffmpeg missing or broken
            print(f"video: imageio-ffmpeg unavailable ({exc}); writing PNG frames instead")
            self.writer = None
            self.frame_dir = self.path.with_suffix("")
            self.frame_dir.mkdir(parents=True, exist_ok=True)

    def append(self, surface) -> None:
        import pygame

        if self.writer is not None:
            self.writer.append_data(pygame.surfarray.array3d(surface).transpose(1, 0, 2))
        else:
            pygame.image.save(surface, str(self.frame_dir / f"{self.count:05d}.png"))
        self.count += 1

    def finish(self, final_surface, hold_s: float = 1.5) -> None:
        for _ in range(int(self.fps * hold_s)):
            self.append(final_surface)
        if self.writer is not None:
            self.writer.close()


class _GifWriter:
    """Looping animated GIF: frames scaled to `width` px with an adaptive 256-colour palette
    each (no dithering, which suits the flat UI and keeps the file small)."""

    def __init__(self, path: str, fps: float, width: int):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.frame_ms = round(1000 / fps)
        self.width = width
        self.frames = []

    def append(self, surface) -> None:
        import pygame
        from PIL import Image

        img = Image.fromarray(pygame.surfarray.array3d(surface).transpose(1, 0, 2))
        if self.width and img.width != self.width:
            img = img.resize((self.width, round(img.height * self.width / img.width)), Image.LANCZOS)
        self.frames.append(img.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE))

    def finish(self, final_surface, hold_s: float = 2.5) -> None:
        self.append(final_surface)
        durations = [self.frame_ms] * (len(self.frames) - 1) + [round(1000 * hold_s)]
        self.frames[0].save(self.path, save_all=True, append_images=self.frames[1:], duration=durations,
                            loop=0, optimize=True)


# ---------------------------------------------------------------------------- viewer
def cmd_viewer(args) -> int:
    import mujoco
    import mujoco.viewer

    env = ComputerUseEnv(env_config_for(args))
    task = make_task(args.task, args.text)
    policy = make_policy(args, env)
    obs = env.reset(task)
    policy.reset()
    try:
        viewer = mujoco.viewer.launch_passive(env.model, env.data)
    except RuntimeError as exc:
        print(f"{exc}\nOn macOS run the viewer with mjpython:  mjpython main.py viewer ...")
        return 1
    print(f"Task: {task.instruction}. Close the viewer window to exit.")
    shown_version = -1
    with viewer:
        viewer.cam.lookat[:] = (0.55, -0.05, 0.95)
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.6, 200, -25
        done = False
        while viewer.is_running():
            start = time.time()
            if not done:
                with viewer.lock():
                    obs, _, done, _ = env.step(policy.act(obs))
                if done:
                    _print_result(env, env.metrics(), 0.0)
            if env.screen.version != shown_version:
                viewer.update_texture(env.screen.id)
                shown_version = env.screen.version
            viewer.sync()
            time.sleep(max(0.0, env.config.control_dt / args.speed - (time.time() - start)))
    return 0


# -------------------------------------------------------------------------- annotate
def cmd_annotate(args) -> int:
    import json

    import numpy as np
    from PIL import Image

    from environment.annotations import draw_annotations
    from learning.scripted_policy import ScriptedPolicy

    env = ComputerUseEnv(EnvConfig(render_obs=False))
    env.reset(make_task(args.task, args.text))
    policy = ScriptedPolicy(env)
    for _ in range(args.steps):  # optionally advance the expert to a more interesting state
        env.step(policy.act())
    height, width = (args.height, args.width) if args.height and args.width else env.camera_size(args.camera)
    image = env.render(args.camera, height, width)
    records = env.annotations(args.camera, height, width)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate([image, draw_annotations(image, records)], axis=1)).save(out)
    visible = [r for r in records if r["visible"] >= 0.5]
    print(f"{args.camera} ({height}x{width}): {len(visible)} of {len(records)} components visible")
    for kind in sorted({r["kind"] for r in visible}):
        labels = [r["label"] for r in visible if r["kind"] == kind]
        print(f"  {kind:13s} {len(labels):3d}  {', '.join(map(repr, labels[:12]))}{' ...' if len(labels) > 12 else ''}")
    print(f"saved {out} (left: camera image, right: with ground-truth boxes)")
    if args.json:
        Path(args.json).write_text(json.dumps(records, indent=1))
        print(f"saved {args.json}")
    env.close()
    return 0


# ---------------------------------------------------------------------- learning loop
def cmd_collect(args) -> int:
    from evaluation.task_success import task_suite
    from learning.dataset import collect_demonstrations

    env = ComputerUseEnv(env_config_for(args))
    tasks = task_suite(args.tasks, args.split, args.episodes, args.seed)
    print(f"Collecting {len(tasks)} expert episodes ({args.split} split, noise {args.noise})")
    t0 = time.time()
    demos = collect_demonstrations(env, tasks, noise=args.noise, perturb_prob=args.perturb_prob, seed=args.seed,
                                   with_images=args.images)
    demos.save(args.out)
    print(f"saved {len(demos)} samples from {len(demos.task_specs)}/{len(tasks)} successful episodes "
          f"to {args.out} in {time.time() - t0:.1f}s")
    return 0


def cmd_train(args) -> int:
    from learning.dataset import Demonstrations
    from learning.policy import train_bc

    demos = Demonstrations.concatenate([Demonstrations.load(p) for p in args.data])
    print(f"Training BC on {len(demos)} samples from {len(demos.task_specs)} episodes")
    train_bc(demos, args.out, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
             use_images=args.images, device=args.device, seed=args.seed)
    return 0


def cmd_evaluate(args) -> int:
    from evaluation.metrics import format_table, save_results, summarize_by
    from evaluation.task_success import evaluate, task_suite

    env = ComputerUseEnv(env_config_for(args))
    policy = make_policy(args, env)
    splits = ("train", "test") if args.split == "both" else (args.split,)
    results = []
    for split in splits:
        print(f"[{split}]")
        tasks = task_suite(args.tasks, split, args.episodes, args.seed)
        results += evaluate(env, policy, tasks, split=split)
    print()
    print(format_table(summarize_by(results, "task", "split"), ("task", "split")))
    if args.out:
        save_results(results, args.out)
        print(f"\nsaved {args.out}")
    return 0


# ------------------------------------------------------------------------------ CLI
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command")

    def task_args(p):
        p.add_argument("--task", choices=ALL_TASKS, default="type_key")
        p.add_argument("--text", help="text / query / app name (default: first test-split item)")

    p = sub.add_parser("demo", help="run one task, optionally with a live dashboard or video")
    task_args(p)
    add_policy_args(p)
    p.add_argument("--display", action="store_true", help="live dashboard window")
    p.add_argument("--video", help="write the dashboard to an .mp4")
    p.add_argument("--gif", help="write the dashboard to an animated .gif")
    p.add_argument("--gif-width", type=int, default=800, help="GIF width in pixels")
    p.add_argument("--gif-fps", type=float, default=10.0, help="GIF frame rate")
    p.add_argument("--screenshot", help="save the final dashboard frame as .png")
    p.add_argument("--camera", default="overview", help="main 3D view camera")
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--speed", type=float, default=1.0, help="playback speed for --display, --video and --gif")
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("viewer", help="interactive MuJoCo viewer (mjpython on macOS)")
    task_args(p)
    add_policy_args(p)
    p.add_argument("--speed", type=float, default=1.0)
    p.set_defaults(func=cmd_viewer)

    p = sub.add_parser("annotate", help="render a camera with ground-truth boxes and labels")
    task_args(p)
    p.add_argument("--camera", default="keyboard_cam",
                   help="robot_cam, keyboard_cam, mouse_cam, screen_cam, wrist_cam, overview, ...")
    p.add_argument("--height", type=int, help="image height (default: the camera's observation size)")
    p.add_argument("--width", type=int)
    p.add_argument("--steps", type=int, default=0, help="expert steps to run before rendering")
    p.add_argument("--out", default="outputs/annotated.png")
    p.add_argument("--json", help="also write the annotation records as JSON")
    p.set_defaults(func=cmd_annotate)

    p = sub.add_parser("collect", help="collect scripted demonstrations")
    p.add_argument("--tasks", nargs="+", choices=ALL_TASKS, default=ALL_TASKS)
    p.add_argument("--split", choices=("train", "test"), default="train")
    p.add_argument("--episodes", type=int, default=30, help="episodes per task")
    p.add_argument("--noise", type=float, default=0.5, help="perturbation size (fraction of max step)")
    p.add_argument("--perturb-prob", type=float, default=0.03, help="per-step chance of a perturbation")
    p.add_argument("--images", action="store_true", help="also record 64x64 camera + screen images")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="data/demos.npz")
    p.set_defaults(func=cmd_collect)

    p = sub.add_parser("train", help="train a behaviour cloning policy")
    p.add_argument("--data", nargs="+", default=["data/demos.npz"])
    p.add_argument("--out", default="checkpoints/bc.pt")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--images", action="store_true", help="use image observations (CNN encoders)")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate", help="success rate, time, failed presses/clicks, collisions")
    add_policy_args(p)
    p.add_argument("--tasks", nargs="+", choices=ALL_TASKS, default=ALL_TASKS)
    p.add_argument("--split", choices=("train", "test", "both"), default="test")
    p.add_argument("--episodes", type=int, default=10, help="episodes per task and split")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", help="write per-episode results + summary as JSON")
    p.set_defaults(func=cmd_evaluate)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:  # milestone 1
        args = parser.parse_args(["demo", "--task", "type_key", "--text", "a"])
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
