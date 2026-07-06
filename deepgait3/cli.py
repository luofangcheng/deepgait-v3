"""CLI entry point for ``deepgait3``.

Subcommands
-----------
  deepgait3 stage1 MOUSE_DIRS ...   Stage 1 footprint extraction
  deepgait3 gui                      Launch the PySide6 GUI (default mode)

The GUI subcommand hosts the merged 11-tab application that replaces
the Sprint 1+2 scaffolding discarded earlier.
"""

from __future__ import annotations

import argparse
from pathlib import Path


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
            tau_paw=args.tau_paw,
            roi_pad=args.roi_pad,
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="deepgait3",
        description="DeepGait v3 - gait analysis platform",
    )
    sub = parser.add_subparsers(dest="command")

    p_gui = sub.add_parser("gui", help="Launch the PySide6 GUI")
    p_gui.add_argument("--debug", action="store_true",
                       help="Enable real-time debug event logging")
    p_gui.add_argument("--log", default=None, metavar="PATH",
                       help="Write debug log to file (requires --debug)")
    p_gui.add_argument("--quiet", action="store_true",
                       help="Suppress stderr debug output (use with --log)")
    p_gui.set_defaults(func=_cmd_gui)

    p1 = sub.add_parser("stage1", help="Stage 1: footprint extraction")
    p1.add_argument("mouse_dirs", nargs="+", help="Mouse directories (one per mouse)")
    p1.add_argument("--tau-paw", type=float, default=10.0,
                    help="Paw delta threshold (default: 10)")
    p1.add_argument("--roi-pad", type=int, default=50,
                    help="Pixels to expand mouse ROI (default: 50)")
    p1.add_argument("--fps", type=float, default=60.0,
                    help="Video frame rate (default: 60)")
    p1.add_argument("--px-per-mm", type=float, default=1.92,
                    help="Pixels-per-mm calibration (default: 1.92)")
    p1.add_argument("--verbose", action="store_true",
                    help="Show full traceback on error")
    p1.set_defaults(func=_cmd_stage1)

    args = parser.parse_args(argv)
    if args.command is None:
        # Default action: launch the GUI.
        return _cmd_gui(args)
    func = getattr(args, "func", None)
    if func is None:
        parser.print_help()
        return 0
    return func(args)


if __name__ == "__main__":
    raise SystemExit(main())
