This folder keeps lightweight model documentation only. Trained model binaries are ignored by Git.

For the backend, place the gesture classifier and MediaPipe hand landmarker at:

```bash
models/WaveSlideV1.tflite
models/hand_landmarker.task
```

The current backend expects the notebook model contract:

- TFLite gesture model
- Input: one RGB batch shaped `(1, 224, 224, 3)`
- Labels by output index: `call`, `fist`, `like`, `two_up`, `unknown`
- Output: softmax probabilities for the five labels
