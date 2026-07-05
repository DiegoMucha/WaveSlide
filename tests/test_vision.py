import numpy as np

from waveslide.vision import BoundingBox, crop_from_bbox, resize_with_padding


def test_crop_from_bbox_uses_normalized_coordinates():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    bbox = BoundingBox(x=0.25, y=0.20, width=0.50, height=0.40)

    crop = crop_from_bbox(image, bbox, margin=0.0)

    assert crop is not None
    assert crop.shape == (40, 100, 3)


def test_resize_with_padding_returns_square_model_input():
    image = np.ones((20, 40, 3), dtype=np.uint8)

    resized = resize_with_padding(image, target_size=224)

    assert resized.shape == (224, 224, 3)
