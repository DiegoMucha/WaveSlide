from __future__ import annotations

import argparse
from pathlib import Path

from waveslide.config import EngineConfig
from waveslide.engine import GesturePresentationEngine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the WaveSlide gesture backend.")
    parser.add_argument("--model", type=Path, default=Path("models/gesture_model.keras"))
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--no-control", action="store_true")
    parser.add_argument("--confidence", type=float, default=0.70)
    parser.add_argument("--stable-frames", type=int, default=4)
    parser.add_argument("--cooldown", type=float, default=1.20)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = EngineConfig(
        model_path=args.model,
        camera_index=args.camera,
        simulate=args.simulate,
        control_presentation=not args.no_control,
        min_prediction_confidence=args.confidence,
        stable_frames=args.stable_frames,
        action_cooldown_seconds=args.cooldown,
    )
    GesturePresentationEngine(config).run(preview=args.preview)


if __name__ == "__main__":
    main()
