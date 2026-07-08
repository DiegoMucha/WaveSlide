from __future__ import annotations

import json
import random
import shutil
from collections import defaultdict
from pathlib import Path


# Edit these values directly.
RAW_IMAGES_ROOT = Path("data/raw/images")
RAW_ANNOTATIONS_ROOT = Path("data/raw/annotations")
PROCESSED_ROOT = Path("data/processed")

MAIN_CLASSES = ("call", "fist", "like", "two_up")
UNKNOWN_CLASS = "unknown"

# None means: use the average number of images in the existing processed main
# classes, so unknown is balanced with the current training folders.
TARGET_COUNT: int | None = None
RANDOM_SEED = 42

# Rebuilds data/raw/images/unknown and data/raw/annotations/unknown.json.
REBUILD_OUTPUT = True

# Match the EDA notebook filtering: only keep images with exactly one bbox.
REQUIRE_SINGLE_BBOX = True

# Your added non-main classes may only have images and no annotation JSON.
# In that case, this writes a full-image bbox: [0, 0, 1, 1].
ASSUME_FULL_IMAGE_BBOX_FOR_UNANNOTATED_CLASSES = True


IMAGE_EXTENSIONS = (".jpg", ".jpeg")


def main() -> None:
    random.seed(RANDOM_SEED)

    target_count = TARGET_COUNT or infer_target_count()
    candidates_by_class = load_unknown_candidates()
    selected = balanced_sample(candidates_by_class, target_count)

    output_images_dir = RAW_IMAGES_ROOT / UNKNOWN_CLASS
    output_annotation_path = RAW_ANNOTATIONS_ROOT / f"{UNKNOWN_CLASS}.json"

    if REBUILD_OUTPUT:
        rebuild_output_dir(output_images_dir)
    else:
        output_images_dir.mkdir(parents=True, exist_ok=True)

    RAW_ANNOTATIONS_ROOT.mkdir(parents=True, exist_ok=True)

    unknown_annotations = {}
    copied = 0
    for item in selected:
        source_class = item["source_class"]
        source_image_id = item["image_id"]
        source_path = item["image_path"]
        annotation = item["annotation"]

        new_image_id = f"{source_class}__{source_image_id}"
        output_path = output_images_dir / f"{new_image_id}.jpg"

        shutil.copy2(source_path, output_path)
        unknown_annotations[new_image_id] = {
            "bboxes": annotation["bboxes"],
            "labels": [UNKNOWN_CLASS for _ in annotation["bboxes"]],
        }
        copied += 1

    output_annotation_path.write_text(
        json.dumps(unknown_annotations, indent=2),
        encoding="utf-8",
    )

    print(f"Target unknown images: {target_count}")
    print(f"Available candidate images: {sum(len(v) for v in candidates_by_class.values())}")
    print(f"Copied unknown images: {copied}")
    print(f"Images written to: {output_images_dir}")
    print(f"Annotations written to: {output_annotation_path}")
    print("Selected source classes:")
    for class_name, count in selected_counts(selected).items():
        print(f"  {class_name}: {count}")


def infer_target_count() -> int:
    processed_counts = [
        count_images(PROCESSED_ROOT / class_name)
        for class_name in MAIN_CLASSES
    ]
    processed_counts = [count for count in processed_counts if count > 0]
    if processed_counts:
        return round(sum(processed_counts) / len(processed_counts))

    raw_counts = [
        len(load_class_candidates(class_name))
        for class_name in MAIN_CLASSES
    ]
    raw_counts = [count for count in raw_counts if count > 0]
    if raw_counts:
        return round(sum(raw_counts) / len(raw_counts))

    raise RuntimeError("Could not infer target count from processed or raw main classes.")


def load_unknown_candidates() -> dict[str, list[dict]]:
    candidates_by_class: dict[str, list[dict]] = {}
    excluded = set(MAIN_CLASSES) | {UNKNOWN_CLASS}

    for class_dir in sorted(RAW_IMAGES_ROOT.iterdir()):
        if not class_dir.is_dir() or class_dir.name in excluded:
            continue

        candidates = load_class_candidates(class_dir.name)
        if candidates:
            candidates_by_class[class_dir.name] = candidates

    if not candidates_by_class:
        raise RuntimeError(
            "No annotated non-main classes found under data/raw/images. "
            "Add extra class folders and matching JSON files in data/raw/annotations first."
        )

    return candidates_by_class


def load_class_candidates(class_name: str) -> list[dict]:
    image_dir = RAW_IMAGES_ROOT / class_name
    annotation_path = RAW_ANNOTATIONS_ROOT / f"{class_name}.json"
    if not image_dir.exists():
        return []

    if not annotation_path.exists():
        if not ASSUME_FULL_IMAGE_BBOX_FOR_UNANNOTATED_CLASSES:
            return []
        return load_image_only_candidates(class_name, image_dir)

    annotations = json.loads(annotation_path.read_text(encoding="utf-8"))
    candidates = []
    for image_id, annotation in annotations.items():
        bboxes = annotation.get("bboxes", [])
        if REQUIRE_SINGLE_BBOX and len(bboxes) != 1:
            continue

        image_path = find_image_path(image_dir, image_id)
        if image_path is None:
            continue

        candidates.append(
            {
                "source_class": class_name,
                "image_id": image_id,
                "image_path": image_path,
                "annotation": annotation,
            }
        )

    return candidates


def load_image_only_candidates(class_name: str, image_dir: Path) -> list[dict]:
    candidates = []
    for image_path in sorted(image_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        candidates.append(
            {
                "source_class": class_name,
                "image_id": image_path.stem,
                "image_path": image_path,
                "annotation": {
                    "bboxes": [[0.0, 0.0, 1.0, 1.0]],
                    "labels": [class_name],
                },
            }
        )

    return candidates


def find_image_path(image_dir: Path, image_id: str) -> Path | None:
    for extension in IMAGE_EXTENSIONS:
        image_path = image_dir / f"{image_id}{extension}"
        if image_path.exists():
            return image_path
    return None


def balanced_sample(
    candidates_by_class: dict[str, list[dict]],
    target_count: int,
) -> list[dict]:
    pools = {
        class_name: random.sample(candidates, len(candidates))
        for class_name, candidates in candidates_by_class.items()
    }
    class_names = sorted(pools)
    per_class = target_count // len(class_names)
    remainder = target_count % len(class_names)

    selected = []
    deficits = 0
    for index, class_name in enumerate(class_names):
        requested = per_class + (1 if index < remainder else 0)
        available = len(pools[class_name])
        take = min(requested, available)
        selected.extend(pools[class_name][:take])
        pools[class_name] = pools[class_name][take:]
        deficits += requested - take

    while deficits > 0:
        progressed = False
        for class_name in class_names:
            if deficits <= 0:
                break
            if not pools[class_name]:
                continue
            selected.append(pools[class_name].pop(0))
            deficits -= 1
            progressed = True
        if not progressed:
            break

    random.shuffle(selected)
    return selected


def count_images(directory: Path) -> int:
    if not directory.exists():
        return 0

    return sum(
        1
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def rebuild_output_dir(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for path in directory.iterdir():
        if path.is_file():
            path.unlink()


def selected_counts(selected: list[dict]) -> dict[str, int]:
    counts = defaultdict(int)
    for item in selected:
        counts[item["source_class"]] += 1
    return dict(sorted(counts.items()))


if __name__ == "__main__":
    main()
