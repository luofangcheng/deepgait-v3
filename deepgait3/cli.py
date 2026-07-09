"""CLI entry point for ``deepgait3``.

Subcommands
-----------
  deepgait3 extract [VIDEO]      Extract footprints from video → active project
  deepgait3 project activate NAME  Set active project
  deepgait3 project list           List projects
  deepgait3 stage1 MOUSE_DIRS      Stage 1 on pre-extracted frames
  deepgait3 gui                    Launch PySide6 GUI
"""
from __future__ import annotations

import argparse
from pathlib import Path


# ── extract (new YOLO pipeline) ──────────────────────────────────────────────

def _cmd_extract(args: argparse.Namespace) -> int:
    from deepgait3.core.data import ProjectManager, extract_trial

    pm = ProjectManager()

    # Auto-activate if there's only one project
    if pm.active_name is None:
        projects = pm.list_projects()
        if len(projects) == 1:
            pm.activate(projects[0])
            print(f"Auto-activated project: {projects[0]}")
        elif not projects:
            print("No projects found. Create one with: deepgait3 project create NAME")
            return 1
        else:
            print(f"Multiple projects: {projects}. Activate one with: "
                  f"deepgait3 project activate NAME")
            return 1

    # Resolve video
    if args.video:
        video_path = Path(args.video).resolve()
    else:
        # Try active project's rawdata/videos
        videos_dir = pm.active_dir / "rawdata" / "videos"
        videos = list(videos_dir.glob("*.mp4")) + list(videos_dir.glob("*.avi"))
        if not videos:
            print(f"No videos in {videos_dir}. Provide --video PATH.")
            return 1
        video_path = videos[0]
        print(f"Using video: {video_path.name}")

    if not video_path.exists():
        print(f"Video not found: {video_path}")
        return 1

    mouse_id = args.mouse_id or video_path.stem
    proj_dir = pm.active_dir
    trial_name = args.name or f"{mouse_id}"

    print(f"Project: {pm.active_name}")
    print(f"Video: {video_path.name}")
    print(f"Mouse: {mouse_id}")
    print(f"px/mm: {args.px_per_mm}")
    print()

    trial = extract_trial(
        video_path=video_path,
        output_dir=proj_dir / "data" / trial_name,
        mouse_id=mouse_id,
        fps=args.fps,
        px_per_mm=args.px_per_mm,
        conf=args.conf,
        min_area_px=args.min_area,
    )

    # Register in project
    pm.save_trial(trial, trial_name=trial_name)
    print(f"\nSaved to: {proj_dir / 'data' / trial_name}")
    return 0


# ── project management ──────────────────────────────────────────────────────

def _cmd_project(args: argparse.Namespace) -> int:
    from deepgait3.core.data import ProjectManager
    pm = ProjectManager()

    if args.project_cmd == "list":
        projects = pm.list_projects()
        active = pm.active_name
        for p in projects:
            marker = " *" if p == active else "  "
            print(f" {marker} {p}")
        if not projects:
            print("No projects. Create: deepgait3 project create NAME")
        return 0

    elif args.project_cmd == "activate":
        pm.activate(args.name)
        print(f"Activated: {args.name}")
        return 0

    elif args.project_cmd == "create":
        pm.create(args.name, experimenter=args.experimenter or "",
                  px_per_mm=args.px_per_mm)
        pm.activate(args.name)
        print(f"Created and activated: {args.name}")
        return 0

    return 0


# ── stage1 (legacy) ─────────────────────────────────────────────────────────

def _cmd_stage1(args: argparse.Namespace) -> int:
    from deepgait3.core.pawprint import Stage1Pipeline

    total = 0
    for mouse_dir in args.mouse_dirs:
        mouse_dir = Path(mouse_dir).resolve()
        if not mouse_dir.is_dir():
            print(f"  SKIP {mouse_dir} - not a directory")
            continue
        mouse_id = mouse_dir.name
        print(f"\n=== {mouse_id} ===")
        pipeline = Stage1Pipeline(
            mouse_id=mouse_id,
            fps=args.fps,
            px_per_mm=args.px_per_mm,
        )
        try:
            pipeline.run(mouse_dir, mouse_dir)
            total += 1
        except Exception as exc:
            print(f"  ERROR [{mouse_id}]: {exc}")
            if args.verbose:
                raise
    print(f"\nDone. {total} trial(s) processed.")
    return 0


# ── gui ─────────────────────────────────────────────────────────────────────

def _cmd_gui(args: argparse.Namespace) -> int:
    from deepgait3.gui.main_window import launch_gui
    argv = []
    if args.debug:
        argv.append("--debug")
        if args.log:
            argv.extend(["--log", args.log])
        if args.quiet:
            argv.append("--quiet")
    return launch_gui(argv if argv else None)


# ── main ────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="deepgait3",
        description="DeepGait v3 - gait analysis platform",
    )
    sub = parser.add_subparsers(dest="command")

    # --- extract ---
    p_ext = sub.add_parser("extract", help="Extract footprints from video → active project")
    p_ext.add_argument("video", nargs="?", default=None,
                       help="Video path (uses active project's rawdata if omitted)")
    p_ext.add_argument("--mouse-id", default="",
                       help="Animal ID (default: video filename)")
    p_ext.add_argument("--name", default="",
                       help="Trial name (default: mouse_id)")
    p_ext.add_argument("--fps", type=float, default=0,
                       help="Frame rate (0=auto-detect)")
    p_ext.add_argument("--px-per-mm", type=float, default=1.92)
    p_ext.add_argument("--conf", type=float, default=0.25,
                       help="YOLO confidence threshold")
    p_ext.add_argument("--min-area", type=int, default=5,
                       help="Minimum footprint area (px)")
    p_ext.set_defaults(func=_cmd_extract)

    # --- project ---
    p_proj = sub.add_parser("project", help="Project management")
    p_proj_sub = p_proj.add_subparsers(dest="project_cmd")

    p_list = p_proj_sub.add_parser("list", help="List projects")
    p_list.set_defaults(func=_cmd_project)

    p_act = p_proj_sub.add_parser("activate", help="Activate a project")
    p_act.add_argument("name", help="Project name")
    p_act.set_defaults(func=_cmd_project)

    p_create = p_proj_sub.add_parser("create", help="Create a new project")
    p_create.add_argument("name", help="Project name")
    p_create.add_argument("--experimenter", default="")
    p_create.add_argument("--px-per-mm", type=float, default=3.0)
    p_create.set_defaults(func=_cmd_project)

    # --- stage1 (legacy, for pre-extracted frame dirs) ---
    p1 = sub.add_parser("stage1", help="Stage 1 on pre-extracted frame dirs")
    p1.add_argument("mouse_dirs", nargs="+", help="Mouse directories")
    p1.add_argument("--fps", type=float, default=60.0)
    p1.add_argument("--px-per-mm", type=float, default=1.92)
    p1.add_argument("--verbose", action="store_true")
    p1.set_defaults(func=_cmd_stage1)

    # --- gui ---
    p_gui = sub.add_parser("gui", help="Launch PySide6 GUI")
    p_gui.add_argument("--debug", action="store_true")
    p_gui.add_argument("--log", default=None, metavar="PATH")
    p_gui.add_argument("--quiet", action="store_true")
    p_gui.set_defaults(func=_cmd_gui)

    args = parser.parse_args(argv)
    if args.command is None:
        return _cmd_gui(args)
    func = getattr(args, "func", None)
    if func is None:
        parser.print_help()
        return 0
    return func(args)


if __name__ == "__main__":
    raise SystemExit(main())
