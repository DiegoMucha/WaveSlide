from __future__ import annotations

import argparse
from pathlib import Path

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
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument("--bbox-margin", type=float, default=0.10)
    parser.add_argument(
        "--mode",
        choices=("direct", "mediapipe"),
        default="direct",
        help="direct uses the full camera frame as the crop; mediapipe crops the detected hand.",
    )
    parser.add_argument("--labels", nargs="+", default=list(DEFAULT_LABELS))
    parser.add_argument("--show-probs", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_realtime_test(
        model_path=args.model,
        camera_index=args.camera,
        threshold=args.threshold,
        bbox_margin=args.bbox_margin,
        mode=args.mode,
        labels=tuple(args.labels),
        show_probs=args.show_probs,
    )


def run_realtime_test(
    model_path: Path,
    camera_index: int,
    threshold: float,
    bbox_margin: float,
    mode: str,
    labels: tuple[str, ...],
    show_probs: bool = False,
) -> None:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required to use the webcam.") from exc

    classifier = TFLiteGestureClassifier(model_path, labels=labels)
    detector = MediaPipeHandDetector() if mode == "mediapipe" else None
    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        if detector is not None:
            detector.close()
        raise RuntimeError(f"Could not open camera index {camera_index}.")

    print("WaveSlide TFLite realtime test")
    print(f"Model: {model_path}")
    print(f"Threshold: {threshold:.2f}")
    print(f"Mode: {mode}")
    print("Press q in the preview window to quit.")

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Could not read a frame from the camera.")

            bbox = None
            crop = frame
            status = "Reading frame"
            if detector is not None:
                status = "No hand"
                bbox = detector.detect(frame)
                if bbox is not None:
                    crop = crop_from_bbox(frame, bbox, margin=bbox_margin)

            if crop is not None:
                model_input = prepare_model_input(crop, target_size=classifier.input_size)
                prediction = classifier.predict(model_input)
                if prediction.confidence >= threshold:
                    status = f"{prediction.label}: {prediction.confidence:.2f}"
                else:
                    status = f"Uncertain: {prediction.label} {prediction.confidence:.2f}"

                if show_probs:
                    print_probabilities(prediction.probabilities)

            cv2.imshow("WaveSlide TFLite Test", draw_status(frame, status, bbox))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        if detector is not None:
            detector.close()
        cv2.destroyAllWindows()


def print_probabilities(probabilities: dict[str, float]) -> None:
    ordered = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    text = " | ".join(f"{label}: {score:.2f}" for label, score in ordered)
    print(text)


if __name__ == "__main__":
    main()
