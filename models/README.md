This folder keeps lightweight model documentation only. Trained model binaries are ignored by Git.

For the backend MVP, place the gesture classifier at:

```bash
models/gesture_model.keras
```

The current backend expects the notebook model contract:

- TensorFlow/Keras model
- Input: one RGB batch shaped `(1, 224, 224, 3)`
- Labels by output index: `call`, `fist`, `like`, `two_up`
- Output: softmax probabilities for the four labels
