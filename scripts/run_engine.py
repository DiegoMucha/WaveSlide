from pathlib import Path

from waveslide.config import DEFAULT_LABELS, EngineConfig
from waveslide.engine import GesturePresentationEngine


# Edit these values directly instead of passing CLI arguments.
MODEL_PATH = Path("models/WaveSlideV1.tflite")
HAND_LANDMARKER_PATH = Path("models/hand_landmarker.task")
CAMERA_INDEX = 0

PREVIEW = True
SIMULATE = False
CONTROL_PRESENTATION = True

MIN_PREDICTION_CONFIDENCE = 0.90
PREDICTION_INTERVAL_MS = 100
STABLE_FRAMES = 4
ACTION_COOLDOWN_SECONDS = 1.20
BBOX_MARGIN = 0.10

LABELS = DEFAULT_LABELS


def main() -> None:
    config = EngineConfig(
        model_path=MODEL_PATH,
        hand_landmarker_path=HAND_LANDMARKER_PATH,
        camera_index=CAMERA_INDEX,
        simulate=SIMULATE,
        control_presentation=CONTROL_PRESENTATION,
        labels=LABELS,
        bbox_margin=BBOX_MARGIN,
        min_prediction_confidence=MIN_PREDICTION_CONFIDENCE,
        prediction_interval_ms=PREDICTION_INTERVAL_MS,
        stable_frames=STABLE_FRAMES,
        action_cooldown_seconds=ACTION_COOLDOWN_SECONDS,
    )
    GesturePresentationEngine(config).run(preview=PREVIEW)


if __name__ == "__main__":
    main()
