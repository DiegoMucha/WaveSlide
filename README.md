# WaveSlide

WaveSlide is a gesture-controlled presentation controller. The backend reads a webcam feed, detects a hand with MediaPipe, crops the hand, runs a TensorFlow gesture classifier, and turns stable gestures into presentation actions.

The frontend lives in `src/frontend` and connects to the backend HTTP/WebSocket API to start/stop the engine and listen for gesture events.

## Installation

Create and activate a Python virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

Install the frontend dependencies:

```bash
cd src/frontend
npm install
cd ../..
```

## Model Setup

The backend expects these model files by default:

```bash
models/WaveSlideV1.tflite
models/hand_landmarker.task
```

## Run The Backend API

Start the backend API from the repository root:

```bash
python scripts/run_api.py
```
Open the API docs at:

```text
http://127.0.0.1:8000/docs
```

## Run The Frontend

Install dependencies and run the Vite dev server from the frontend folder:

```bash
cd src/frontend
npm install
npm run dev
```

The frontend should connect to the backend at:

```text
http://127.0.0.1:8000
```

## Run The Local Engine Directly

Run the engine script without the HTTP API:
```bash
python scripts/run_engine.py
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