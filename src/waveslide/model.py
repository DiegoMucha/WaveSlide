from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from random import Random

import numpy as np

from waveslide.config import DEFAULT_LABELS


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float
    probabilities: dict[str, float]


class GestureClassifier:
    def __init__(self, model_path: Path | str, labels: tuple[str, ...] = DEFAULT_LABELS) -> None:
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise RuntimeError(
                "tensorflow is required to run gesture classification. Install project requirements first."
            ) from exc

        self.labels = labels
        self.model = tf.keras.models.load_model(model_path)

    def predict(self, batch_rgb: np.ndarray) -> Prediction:
        scores = self.model.predict(batch_rgb, verbose=0)[0]
        best_index = int(np.argmax(scores))
        probabilities = {
            label: float(scores[index])
            for index, label in enumerate(self.labels)
            if index < len(scores)
        }
        return Prediction(
            label=self.labels[best_index],
            confidence=float(scores[best_index]),
            probabilities=probabilities,
        )


class TFLiteGestureClassifier:
    def __init__(self, model_path: Path | str, labels: tuple[str, ...] = DEFAULT_LABELS) -> None:
        self.labels = labels
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"TFLite model not found: {self.model_path}")

        self._interpreter = self._create_interpreter()
        self._interpreter.allocate_tensors()
        self._input_details = self._interpreter.get_input_details()[0]
        self._output_details = self._interpreter.get_output_details()[0]

    @property
    def input_size(self) -> int:
        shape = self._input_details["shape"]
        if len(shape) != 4:
            raise ValueError(f"Expected a 4D model input, got shape: {shape}")
        return int(shape[1])

    def predict(self, batch_rgb: np.ndarray) -> Prediction:
        model_input = self._prepare_input(batch_rgb)
        self._interpreter.set_tensor(self._input_details["index"], model_input)
        self._interpreter.invoke()
        output = self._interpreter.get_tensor(self._output_details["index"])[0]
        scores = self._dequantize_output(output)
        return prediction_from_scores(scores, self.labels)

    def _create_interpreter(self):
        try:
            from tflite_runtime.interpreter import Interpreter

            return Interpreter(model_path=str(self.model_path))
        except ImportError:
            try:
                import tensorflow as tf
            except ImportError as exc:
                raise RuntimeError(
                    "Install tensorflow or tflite-runtime to run TFLite inference."
                ) from exc

            return tf.lite.Interpreter(model_path=str(self.model_path))

    def _prepare_input(self, batch_rgb: np.ndarray) -> np.ndarray:
        input_dtype = self._input_details["dtype"]
        if input_dtype == np.float32:
            return batch_rgb.astype(np.float32)

        scale, zero_point = self._input_details.get("quantization", (0.0, 0))
        if scale:
            quantized = batch_rgb / scale + zero_point
            limits = np.iinfo(input_dtype)
            return np.clip(np.rint(quantized), limits.min, limits.max).astype(input_dtype)

        return batch_rgb.astype(input_dtype)

    def _dequantize_output(self, output: np.ndarray) -> np.ndarray:
        output = np.asarray(output)
        if np.issubdtype(output.dtype, np.floating):
            return output.astype(np.float32)

        scale, zero_point = self._output_details.get("quantization", (0.0, 0))
        if scale:
            return (output.astype(np.float32) - zero_point) * scale

        return output.astype(np.float32)


def prediction_from_scores(scores: np.ndarray, labels: tuple[str, ...]) -> Prediction:
    best_index = int(np.argmax(scores))
    probabilities = {
        label: float(scores[index])
        for index, label in enumerate(labels)
        if index < len(scores)
    }
    return Prediction(
        label=labels[best_index],
        confidence=float(scores[best_index]),
        probabilities=probabilities,
    )


class SimulatedGestureClassifier:
    def __init__(self, labels: tuple[str, ...] = DEFAULT_LABELS) -> None:
        self.labels = labels
        self._index = 0
        self._random = Random(42)

    def predict_next(self) -> Prediction:
        label = self.labels[self._index % len(self.labels)]
        self._index += 1
        confidence = 0.82 + self._random.random() * 0.15
        probabilities = {item: 0.02 for item in self.labels}
        probabilities[label] = confidence
        return Prediction(label=label, confidence=confidence, probabilities=probabilities)
