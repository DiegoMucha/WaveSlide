from waveslide.actions import GestureActionDebouncer, execute_action


class FakeController:
    def __init__(self):
        self.calls = []

    def next_slide(self):
        self.calls.append("next_slide")


def test_debouncer_requires_stable_confident_frames():
    debouncer = GestureActionDebouncer(
        gesture_actions={"like": "next_slide"},
        min_confidence=0.70,
        stable_frames=3,
        cooldown_seconds=1.0,
    )

    assert debouncer.update("like", 0.69, now=0.0) is None
    assert debouncer.update("like", 0.95, now=0.1) is None
    assert debouncer.update("like", 0.95, now=0.2) is None
    assert debouncer.update("like", 0.95, now=1.2) == "next_slide"


def test_execute_action_calls_controller_method():
    controller = FakeController()

    execute_action(controller, "next_slide")

    assert controller.calls == ["next_slide"]
