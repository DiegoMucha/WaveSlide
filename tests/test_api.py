from waveslide.api import ConfigUpdate, EngineManager, serialize_config
from waveslide.config import EngineConfig


def test_engine_manager_updates_config_when_stopped():
    manager = EngineManager(EngineConfig(simulate=True))

    config = manager.update_config(
        ConfigUpdate(
            model_path="models/example.keras",
            hand_landmarker_path="models/example.task",
            camera_index=1,
            min_prediction_confidence=0.8,
        )
    )

    assert str(config.model_path) == "models/example.keras"
    assert str(config.hand_landmarker_path) == "models/example.task"
    assert config.camera_index == 1
    assert config.min_prediction_confidence == 0.8


def test_serialize_config_converts_path_to_string():
    payload = serialize_config(EngineConfig())

    assert payload["model_path"] == "models/WaveSlideV1.tflite"
    assert payload["hand_landmarker_path"] == "models/hand_landmarker.task"
