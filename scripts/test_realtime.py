from pathlib import Path

from waveslide.config import DEFAULT_LABELS
from waveslide.tflite_realtime import run_realtime_test


# Edit these values directly instead of passing CLI arguments.
MODEL_PATH = Path("models/WaveSlideV1.tflite")
HAND_LANDMARKER_PATH = Path("models/hand_landmarker.task")
CAMERA_INDEX = 0

THRESHOLD = 0.90
BBOX_MARGIN = 0.10
PREDICTION_INTERVAL_MS = 100
MODE = "mediapipe"
LABELS = DEFAULT_LABELS
SHOW_PROBS_IN_TERMINAL = False


def main() -> None:
    run_realtime_test(
        model_path=MODEL_PATH,
        hand_landmarker_path=HAND_LANDMARKER_PATH,
        camera_index=CAMERA_INDEX,
        threshold=THRESHOLD,
        bbox_margin=BBOX_MARGIN,
        prediction_interval_ms=PREDICTION_INTERVAL_MS,
        mode=MODE,
        labels=LABELS,
        show_probs=SHOW_PROBS_IN_TERMINAL,
    )


if __name__ == "__main__":
    main()
