from __future__ import annotations

from dataclasses import dataclass
from threading import Event
from time import sleep
from typing import Callable

from waveslide.actions import (
    GestureActionDebouncer,
    KeyboardPresentationController,
    execute_action,
)
from waveslide.config import EngineConfig
from waveslide.model import GestureClassifier, Prediction, SimulatedGestureClassifier
from waveslide.vision import MediaPipeHandDetector, crop_from_bbox, draw_status, prepare_model_input


@dataclass(frozen=True)
class EngineEvent:
    type: str
    running: bool
    gesture: str | None = None
    confidence: float | None = None
    action: str | None = None
    message: str | None = None


EngineEventHandler = Callable[[EngineEvent], None]


@dataclass
class GesturePresentationEngine:
    config: EngineConfig

    def run(
        self,
        preview: bool = False,
        stop_event: Event | None = None,
        on_event: EngineEventHandler | None = None,
    ) -> None:
        stop_event = stop_event or Event()
        if self.config.simulate:
            self._run_simulation(stop_event=stop_event, on_event=on_event)
            return

        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("opencv-python is required to read camera frames.") from exc

        detector = MediaPipeHandDetector(
            min_detection_confidence=self.config.min_detection_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
        )
        classifier = GestureClassifier(self.config.model_path, labels=self.config.labels)
        controller = (
            KeyboardPresentationController()
            if self.config.control_presentation
            else None
        )
        debouncer = GestureActionDebouncer(
            gesture_actions=self.config.gesture_actions,
            min_confidence=self.config.min_prediction_confidence,
            stable_frames=self.config.stable_frames,
            cooldown_seconds=self.config.action_cooldown_seconds,
        )

        camera = cv2.VideoCapture(self.config.camera_index)
        if not camera.isOpened():
            detector.close()
            raise RuntimeError(f"Could not open camera index {self.config.camera_index}.")

        try:
            self._emit(on_event, EngineEvent(type="status", running=True, message="engine_started"))
            while not stop_event.is_set():
                ok, frame = camera.read()
                if not ok:
                    raise RuntimeError("Could not read a frame from the camera.")

                status = "No hand"
                bbox = detector.detect(frame)
                if bbox is not None:
                    crop = crop_from_bbox(frame, bbox, margin=self.config.bbox_margin)
                    if crop is not None:
                        model_input = prepare_model_input(crop, target_size=self.config.input_size)
                        prediction = classifier.predict(model_input)
                        status = f"{prediction.label} {prediction.confidence:.2f}"
                        action = debouncer.update(prediction.label, prediction.confidence)
                        if action is not None:
                            if controller is not None:
                                execute_action(controller, action)
                            status = f"{status} -> {action}"
                        self._emit_prediction(on_event, prediction, action)

                if preview:
                    cv2.imshow("WaveSlide", draw_status(frame, status, bbox))
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            camera.release()
            detector.close()
            if preview:
                cv2.destroyAllWindows()
            self._emit(on_event, EngineEvent(type="status", running=False, message="engine_stopped"))

    def _run_simulation(
        self,
        stop_event: Event,
        on_event: EngineEventHandler | None = None,
    ) -> None:
        classifier = SimulatedGestureClassifier(labels=self.config.labels)
        debouncer = GestureActionDebouncer(
            gesture_actions=self.config.gesture_actions,
            min_confidence=self.config.min_prediction_confidence,
            stable_frames=1,
            cooldown_seconds=self.config.action_cooldown_seconds,
        )

        self._emit(on_event, EngineEvent(type="status", running=True, message="simulation_started"))
        try:
            while not stop_event.is_set():
                prediction = classifier.predict_next()
                action = debouncer.update(prediction.label, prediction.confidence)
                self._emit_prediction(on_event, prediction, action)
                sleep(0.75)
        finally:
            self._emit(on_event, EngineEvent(type="status", running=False, message="simulation_stopped"))

    def _emit_prediction(
        self,
        on_event: EngineEventHandler | None,
        prediction: Prediction,
        action: str | None,
    ) -> None:
        self._emit(
            on_event,
            EngineEvent(
                type="prediction",
                running=True,
                gesture=prediction.label,
                confidence=prediction.confidence,
                action=action,
            ),
        )

    def _emit(self, on_event: EngineEventHandler | None, event: EngineEvent) -> None:
        if on_event is not None:
            on_event(event)
