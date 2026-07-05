from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path


DEFAULT_LABELS = ("call", "fist", "like", "two_up")


@dataclass(frozen=True)
class EngineConfig:
    model_path: Path = Path("models/gesture_model.keras")
    camera_index: int = 0
    simulate: bool = False
    control_presentation: bool = True
    labels: tuple[str, ...] = DEFAULT_LABELS
    input_size: int = 224
    bbox_margin: float = 0.10
    min_detection_confidence: float = 0.60
    min_tracking_confidence: float = 0.50
    min_prediction_confidence: float = 0.70
    stable_frames: int = 4
    action_cooldown_seconds: float = 1.20
    gesture_actions: dict[str, str] = field(
        default_factory=lambda: {
            "like": "next_slide",
            "two_up": "previous_slide",
            "fist": "toggle_black_screen",
            "call": "start_presentation",
        }
    )


def config_from_env() -> EngineConfig:
    return EngineConfig(
        model_path=Path(os.getenv("WAVESLIDE_MODEL_PATH", "models/gesture_model.keras")),
        camera_index=int(os.getenv("WAVESLIDE_CAMERA_INDEX", "0")),
        simulate=os.getenv("WAVESLIDE_SIMULATE", "false").lower() in {"1", "true", "yes"},
        control_presentation=os.getenv("WAVESLIDE_CONTROL_PRESENTATION", "true").lower()
        in {"1", "true", "yes"},
        min_prediction_confidence=float(os.getenv("WAVESLIDE_CONFIDENCE", "0.70")),
        stable_frames=int(os.getenv("WAVESLIDE_STABLE_FRAMES", "4")),
        action_cooldown_seconds=float(os.getenv("WAVESLIDE_COOLDOWN_SECONDS", "1.20")),
    )
