from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BoundingBox:
    x: float
    y: float
    width: float
    height: float


class MediaPipeHandDetector:
    def __init__(
        self,
        min_detection_confidence: float = 0.60,
        min_tracking_confidence: float = 0.50,
    ) -> None:
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError(
                "mediapipe is required to detect hands. Install project requirements first."
            ) from exc

        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def detect(self, frame_bgr: np.ndarray) -> BoundingBox | None:
        import cv2

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._hands.process(frame_rgb)
        if not result.multi_hand_landmarks:
            return None

        landmarks = result.multi_hand_landmarks[0].landmark
        xs = [point.x for point in landmarks]
        ys = [point.y for point in landmarks]
        x1 = max(0.0, min(xs))
        y1 = max(0.0, min(ys))
        x2 = min(1.0, max(xs))
        y2 = min(1.0, max(ys))
        return BoundingBox(x=x1, y=y1, width=x2 - x1, height=y2 - y1)

    def close(self) -> None:
        self._hands.close()


def crop_from_bbox(
    image: np.ndarray,
    bbox: BoundingBox,
    margin: float = 0.10,
) -> np.ndarray | None:
    img_h, img_w = image.shape[:2]
    x1 = int(bbox.x * img_w)
    y1 = int(bbox.y * img_h)
    x2 = int((bbox.x + bbox.width) * img_w)
    y2 = int((bbox.y + bbox.height) * img_h)

    bbox_w = x2 - x1
    bbox_h = y2 - y1
    margin_x = int(bbox_w * margin)
    margin_y = int(bbox_h * margin)

    x1 = max(0, x1 - margin_x)
    y1 = max(0, y1 - margin_y)
    x2 = min(img_w, x2 + margin_x)
    y2 = min(img_h, y2 + margin_y)

    if x2 <= x1 or y2 <= y1:
        return None
    return image[y1:y2, x1:x2]


def resize_with_padding(
    image: np.ndarray,
    target_size: int = 224,
    pad_color: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required to preprocess frames.") from exc

    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((target_size, target_size, 3), pad_color, dtype=np.uint8)
    x_offset = (target_size - new_w) // 2
    y_offset = (target_size - new_h) // 2
    canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized
    return canvas


def prepare_model_input(crop_bgr: np.ndarray, target_size: int = 224) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required to preprocess frames.") from exc

    crop_bgr = resize_with_padding(crop_bgr, target_size=target_size)
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    return np.expand_dims(crop_rgb.astype(np.float32), axis=0)


def draw_status(frame: np.ndarray, text: str, bbox: BoundingBox | None = None) -> Any:
    import cv2

    if bbox is not None:
        h, w = frame.shape[:2]
        p1 = (int(bbox.x * w), int(bbox.y * h))
        p2 = (int((bbox.x + bbox.width) * w), int((bbox.y + bbox.height) * h))
        cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)

    cv2.putText(
        frame,
        text,
        (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return frame
