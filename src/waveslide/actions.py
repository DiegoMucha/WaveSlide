from __future__ import annotations

from dataclasses import dataclass
from time import monotonic
from typing import Protocol


class PresentationController(Protocol):
    def next_slide(self) -> None: ...

    def previous_slide(self) -> None: ...

    def start_presentation(self) -> None: ...

    def toggle_black_screen(self) -> None: ...


class KeyboardPresentationController:
    def __init__(self) -> None:
        try:
            import pyautogui
        except ImportError as exc:
            raise RuntimeError(
                "pyautogui is required to control presentations. Install project requirements first."
            ) from exc

        self._keyboard = pyautogui

    def next_slide(self) -> None:
        self._keyboard.press("right")

    def previous_slide(self) -> None:
        self._keyboard.press("left")

    def start_presentation(self) -> None:
        self._keyboard.press("f5")

    def toggle_black_screen(self) -> None:
        self._keyboard.press("b")


@dataclass
class GestureActionDebouncer:
    gesture_actions: dict[str, str]
    min_confidence: float = 0.90
    stable_frames: int = 4
    cooldown_seconds: float = 1.20

    def __post_init__(self) -> None:
        self._last_label: str | None = None
        self._stable_count = 0
        self._last_action_at = 0.0

    def update(self, label: str, confidence: float, now: float | None = None) -> str | None:
        now = monotonic() if now is None else now
        if confidence < self.min_confidence:
            self._last_label = None
            self._stable_count = 0
            return None

        if label == self._last_label:
            self._stable_count += 1
        else:
            self._last_label = label
            self._stable_count = 1

        if self._stable_count < self.stable_frames:
            return None
        if now - self._last_action_at < self.cooldown_seconds:
            return None

        action = self.gesture_actions.get(label)
        if action is None:
            return None

        self._last_action_at = now
        self._stable_count = 0
        return action


def execute_action(controller: PresentationController, action: str) -> None:
    try:
        method = getattr(controller, action)
    except AttributeError as exc:
        raise ValueError(f"Unknown presentation action: {action}") from exc
    method()
