# WaveSlide

WaveSlide is a gesture-controlled presentation controller. The backend reads a webcam feed, detects a hand with MediaPipe, crops the hand, runs a TensorFlow gesture classifier, and turns stable gestures into presentation actions.

The frontend is not implemented yet, but the backend already exposes an HTTP/WebSocket API so a future web UI can start/stop the engine and listen for gesture events.

## Requirements

- Python 3.10+
- A webcam
- A trained TFLite gesture model and MediaPipe hand landmarker for real inference
- Node.js and npm for the future frontend

For development before the model or frontend is ready, use simulation mode.

## Installation

Create and activate a Python virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the project dependencies:

```bash
pip install -r requirements.txt
```

This installs the package in editable mode because `requirements.txt` includes:

```bash
-e .
```

After installation, FastAPI should be available:

```bash
fastapi --help
```

## Model Setup

The backend expects these model files by default:

```bash
models/WaveSlideV1.tflite
models/hand_landmarker.task
```

The current model contract is:

- TFLite gesture classifier
- MediaPipe Tasks hand landmarker used first to create the hand bounding box
- Input: RGB batch shaped `(1, 224, 224, 3)`
- Labels by output index: `call`, `fist`, `like`, `two_up`, `unknown`
- Output: softmax probabilities for the five labels

Real model files are ignored by Git. Keep only lightweight files like `models/.gitkeep` and `models/README.md` in the repo.

## Run The Backend API

Use simulation mode while the model or frontend is not ready:

```bash
WAVESLIDE_SIMULATE=true fastapi dev src/waveslide/api.py
```

Open the API docs at:

```text
http://127.0.0.1:8000/docs
```

Run with the real model later:

```bash
WAVESLIDE_MODEL_PATH=models/WaveSlideV1.tflite fastapi dev src/waveslide/api.py
```

Useful environment variables:

```bash
WAVESLIDE_MODEL_PATH=models/WaveSlideV1.tflite
WAVESLIDE_HAND_LANDMARKER_PATH=models/hand_landmarker.task
WAVESLIDE_SIMULATE=true
WAVESLIDE_CAMERA_INDEX=0
WAVESLIDE_CONFIDENCE=0.90
WAVESLIDE_PREDICTION_INTERVAL_MS=100
WAVESLIDE_STABLE_FRAMES=4
WAVESLIDE_COOLDOWN_SECONDS=1.20
```

## Backend API

| Route | Purpose |
| --- | --- |
| `GET /health` | API and engine health |
| `GET /config` | Current engine config |
| `POST /config` | Update model path, camera, confidence, simulation, or control settings |
| `GET /engine/status` | Running state, last event, and last error |
| `POST /engine/start` | Start the detection loop |
| `POST /engine/stop` | Stop the detection loop |
| `WS /ws/events` | Stream gesture, action, status, and error events |

Example event:

```json
{
  "type": "prediction",
  "running": true,
  "gesture": "like",
  "confidence": 0.91,
  "action": "next_slide",
  "message": null
}
```

## Run The Local Engine Directly

Run the engine script without the HTTP API:

```bash
python scripts/run_engine.py
```

Edit the constants at the top of `scripts/run_engine.py` to change model path, camera, threshold, prediction interval, and preview settings.

Press `q` in the preview window to stop.

Use simulation mode without camera/model dependencies by setting this in `scripts/run_engine.py`:

```python
SIMULATE = True
```

Disable keyboard control while testing by setting:

```python
CONTROL_PRESENTATION = False
```

## Test A TFLite Model In Real Time

Place your TFLite model in `models/`. The default test command expects:

```bash
models/WaveSlideV1.tflite
```

Run the webcam tester:

```bash
python scripts/test_realtime.py
```

By default, this uses `mediapipe` mode: the camera frame goes through `models/hand_landmarker.task`, the hand landmarks are converted into a bounding box, the crop is padded to 224x224, converted from BGR to RGB, converted to `float32`, and then sent to the TFLite model. Detection and classification run every 100 ms by default so the camera preview stays responsive.

The preview window shows the top gesture only when its probability is at least `0.90`. If the best prediction is below that threshold, it shows `Uncertain`.

Show all class probabilities in the terminal by setting this in `scripts/test_realtime.py`:

```python
SHOW_PROBS_IN_TERMINAL = True
```

To bypass MediaPipe temporarily and test the classifier on the full frame, set:

```python
MODE = "direct"
```

Press `q` in the preview window to stop.

## Gesture Actions

Default gesture mapping:

| Gesture | Action |
| --- | --- |
| `like` | Next slide |
| `two_up` | Previous slide |
| `fist` | Toggle black screen |
| `call` | Start presentation |

## Future Frontend

The frontend is expected to be a Node/npm app. Once it exists, install and run it from the frontend folder:

```bash
cd frontend
npm install
npm run dev
```

The frontend should connect to the backend at:

```text
http://127.0.0.1:8000
ws://127.0.0.1:8000/ws/events
```

For development, start the backend API first, then start the frontend dev server.

## Tests

Run tests with:

```bash
pytest
```

If `pytest` is not available, install the dependencies again inside the active virtual environment:

```bash
pip install -r requirements.txt
```

## Project Organization

```text
├── data
│   ├── interim        <- Intermediate data, ignored except .gitkeep
│   ├── processed      <- Processed data, ignored except .gitkeep
│   └── raw            <- Raw data, ignored except .gitkeep
├── docs               <- Documentation project
├── models             <- Local trained models, ignored except docs/placeholders
├── notebooks          <- EDA, modeling, and evaluation notebooks
├── references         <- Data dictionaries and supporting references
├── reports            <- Generated reports and figures
├── src/waveslide      <- Backend package
├── tests              <- Unit tests
├── pyproject.toml     <- Package metadata and CLI entrypoints
├── requirements.txt   <- Python dependencies
└── README.md
```
