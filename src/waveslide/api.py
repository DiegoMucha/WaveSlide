from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, replace
from pathlib import Path
from queue import SimpleQueue
from threading import Event, Lock, Thread
from time import monotonic
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import numpy as np
from pydantic import BaseModel, Field

from waveslide.config import EngineConfig, config_from_env
from waveslide.engine import EngineEvent, GesturePresentationEngine


class ConfigUpdate(BaseModel):
    model_path: str | None = None
    hand_landmarker_path: str | None = None
    camera_index: int | None = None
    simulate: bool | None = None
    control_presentation: bool | None = None
    min_prediction_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    prediction_interval_ms: int | None = Field(default=None, ge=0)
    stable_frames: int | None = Field(default=None, ge=1)
    action_cooldown_seconds: float | None = Field(default=None, ge=0.0)


class EngineManager:
    def __init__(self, config: EngineConfig) -> None:
        self._config = config
        self._thread: Thread | None = None
        self._stop_event: Event | None = None
        self._lock = Lock()
        self._subscribers: list[SimpleQueue[dict[str, Any]]] = []
        self._last_event: dict[str, Any] | None = None
        self._last_error: str | None = None
        self._latest_frame_jpeg: bytes | None = None
        self._last_frame_encoded_at = 0.0
        self._frame_lock = Lock()

    @property
    def config(self) -> EngineConfig:
        return self._config

    def status(self) -> dict[str, Any]:
        return {
            "running": self.is_running,
            "last_event": self._last_event,
            "last_error": self._last_error,
        }

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def update_config(self, update: ConfigUpdate) -> EngineConfig:
        if self.is_running:
            raise RuntimeError("Stop the engine before changing config.")

        changes: dict[str, Any] = {}
        update_payload = (
            update.model_dump(exclude_unset=True)
            if hasattr(update, "model_dump")
            else update.dict(exclude_unset=True)
        )

        for field, value in update_payload.items():
            if value is None:
                continue
            changes[field] = (
                Path(value)
                if field in {"model_path", "hand_landmarker_path"}
                else value
            )

        self._config = replace(self._config, **changes)
        return self._config

    def start(self) -> None:
        with self._lock:
            if self.is_running:
                return

            self._last_error = None
            self._stop_event = Event()
            self._thread = Thread(target=self._run_engine, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        with self._lock:
            if self._stop_event is not None:
                self._stop_event.set()

    def subscribe(self) -> SimpleQueue[dict[str, Any]]:
        queue: SimpleQueue[dict[str, Any]] = SimpleQueue()
        self._subscribers.append(queue)
        return queue

    def unsubscribe(self, queue: SimpleQueue[dict[str, Any]]) -> None:
        if queue in self._subscribers:
            self._subscribers.remove(queue)

    def _run_engine(self) -> None:
        try:
            GesturePresentationEngine(self._config).run(
                stop_event=self._stop_event,
                on_event=self._publish,
                on_frame=self._publish_frame,
            )
        except Exception as exc:
            self._last_error = str(exc)
            self._publish(EngineEvent(type="error", running=False, message=str(exc)))

    def _publish(self, event: EngineEvent) -> None:
        payload = asdict(event)
        self._last_event = payload
        for queue in list(self._subscribers):
            queue.put(payload)

    def _publish_frame(self, frame: np.ndarray) -> None:
        now = monotonic()
        if now - self._last_frame_encoded_at < 1 / 30:
            return
        self._last_frame_encoded_at = now

        try:
            import cv2
        except ImportError:
            return

        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            return

        with self._frame_lock:
            self._latest_frame_jpeg = encoded.tobytes()

    def latest_frame(self) -> bytes | None:
        with self._frame_lock:
            return self._latest_frame_jpeg


def serialize_config(config: EngineConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["model_path"] = str(config.model_path)
    payload["hand_landmarker_path"] = str(config.hand_landmarker_path)
    return payload


def create_app(config: EngineConfig | None = None) -> FastAPI:
    manager = EngineManager(config or config_from_env())
    app = FastAPI(title="WaveSlide API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://127.0.0.1:5173",
            "http://localhost:5173",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.manager = manager

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"ok": True, **manager.status()}

    @app.get("/config")
    def get_config() -> dict[str, Any]:
        return serialize_config(manager.config)

    @app.post("/config")
    def update_config(update: ConfigUpdate) -> dict[str, Any]:
        try:
            config = manager.update_config(update)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return serialize_config(config)

    @app.get("/engine/status")
    def engine_status() -> dict[str, Any]:
        return manager.status()

    @app.get("/video/stream")
    def video_stream() -> StreamingResponse:
        return StreamingResponse(
            stream_frames(manager),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @app.post("/engine/start")
    def start_engine() -> dict[str, Any]:
        manager.start()
        return manager.status()

    @app.post("/engine/stop")
    def stop_engine() -> dict[str, Any]:
        manager.stop()
        return manager.status()

    @app.websocket("/ws/events")
    async def events(websocket: WebSocket) -> None:
        await websocket.accept()
        queue = manager.subscribe()
        try:
            while True:
                payload = await asyncio.to_thread(queue.get)
                await websocket.send_json(payload)
        except WebSocketDisconnect:
            manager.unsubscribe(queue)

    return app


async def stream_frames(manager: EngineManager):
    while True:
        frame = manager.latest_frame()
        if frame is not None:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Cache-Control: no-cache\r\n\r\n"
                + frame
                + b"\r\n"
            )
        await asyncio.sleep(0.05)


app = create_app()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the WaveSlide HTTP/WebSocket API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--hand-landmarker", type=Path, default=None)
    args = parser.parse_args()

    if args.simulate or args.model is not None or args.hand_landmarker is not None:
        env_config = config_from_env()
        config = replace(
            env_config,
            simulate=args.simulate,
            model_path=args.model or env_config.model_path,
            hand_landmarker_path=args.hand_landmarker
            or env_config.hand_landmarker_path,
        )
        global app
        app = create_app(config)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
