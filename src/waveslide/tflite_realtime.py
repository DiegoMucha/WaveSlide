from __future__ import annotations

import argparse
from pathlib import Path
from time import monotonic

from waveslide.config import DEFAULT_LABELS
from waveslide.model import TFLiteGestureClassifier
from waveslide.vision import (
    MediaPipeHandDetector,
    crop_from_bbox,
    draw_status,
    prepare_model_input,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Test a WaveSlide TFLite gesture model with the webcam."
    )
    parser.add_argument("--model", type=Path, default=Path("models/WaveSlideV1.tflite"))
    parser.add_argument(
        "--hand-landmarker",
        type=Path,
        default=Path("models/hand_landmarker.task"),
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--bbox-margin", type=float, default=0.10)
    parser.add_argument("--prediction-interval-ms", type=int, default=100)
    parser.add_argument(
        "--mode",
        choices=("direct", "mediapipe"),
        default="mediapipe",
        help="direct uses the full camera frame as the crop; mediapipe crops the detected hand.",
    )
    parser.add_argument("--labels", nargs="+", default=list(DEFAULT_LABELS))
    parser.add_argument("--show-probs", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_realtime_test(
        model_path=args.model,
        hand_landmarker_path=args.hand_landmarker,
        camera_index=args.camera,
        threshold=args.threshold,
        bbox_margin=args.bbox_margin,
        prediction_interval_ms=args.prediction_interval_ms,
        mode=args.mode,
        labels=tuple(args.labels),
        show_probs=args.show_probs,
    )


def run_realtime_test(
    model_path: Path,
    hand_landmarker_path: Path,
    camera_index: int,
    threshold: float,
    bbox_margin: float,
    prediction_interval_ms: int,
    mode: str,
    labels: tuple[str, ...],
    show_probs: bool = False,
) -> None:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required to use the webcam.") from exc

    classifier = TFLiteGestureClassifier(model_path, labels=labels)
    detector = (
        MediaPipeHandDetector(hand_landmarker_path=hand_landmarker_path)
        if mode == "mediapipe"
        else None
    )
    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        if detector is not None:
            detector.close()
        raise RuntimeError(f"Could not open camera index {camera_index}.")

    print("WaveSlide TFLite realtime test")
    print(f"Model: {model_path}")
    if detector is not None:
        print(f"Hand landmarker: {hand_landmarker_path}")
    print(f"Threshold: {threshold:.4f}")
    print(f"Prediction interval: {prediction_interval_ms} ms")
    print(f"Mode: {mode}")
    print("Press q in the preview window to quit.")

    prediction_interval_seconds = max(0, prediction_interval_ms) / 1000
    last_prediction_at = 0.0
    bbox = None
    status = "Starting"
    probability_line = ""

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Could not read a frame from the camera.")

            now = monotonic()
            should_predict = (
                prediction_interval_seconds == 0
                or now - last_prediction_at >= prediction_interval_seconds
            )
            if should_predict:
                last_prediction_at = now
                crop = frame if detector is None else None
                status = "Reading frame"
                if detector is not None:
                    status = "No hand"
                    probability_line = ""
                    bbox = detector.detect(frame)
                    if bbox is not None:
                        crop = crop_from_bbox(frame, bbox, margin=bbox_margin)

                if crop is not None:
                    model_input = prepare_model_input(crop, target_size=classifier.input_size)
                    prediction = classifier.predict(model_input)
                    probability_line = format_probabilities(prediction.probabilities, labels)
                    if prediction.confidence >= threshold:
                        status = f"{prediction.label}: {prediction.confidence:.4f}"
                    else:
                        status = f"Uncertain: {prediction.label} {prediction.confidence:.4f}"

                    if show_probs:
                        print_probabilities(prediction.probabilities)

            details = (probability_line,) if probability_line else ()
            cv2.imshow("WaveSlide TFLite Test", draw_status(frame, status, bbox, details))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        if detector is not None:
            detector.close()
        cv2.destroyAllWindows()


def print_probabilities(probabilities: dict[str, float]) -> None:
    ordered = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    text = " | ".join(f"{label}: {score:.4f}" for label, score in ordered)
    print(text)


def format_probabilities(
    probabilities: dict[str, float],
    labels: tuple[str, ...],
) -> str:
    return " | ".join(f"{label}: {probabilities.get(label, 0.0):.4f}" for label in labels)


if __name__ == "__main__":
    main()
