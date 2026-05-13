#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

import convert_labelme_to_curvelanes as conv


THREE_CLASS_COLOR_BGR = {
    "egoleft": (80, 220, 80),
    "egoright": (255, 200, 0),
    "other_lanes": (255, 120, 120),
}

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
DEFAULT_DOWNSAMPLE_FACTOR = 4


def configure_converter(class_mode: int, include_stop_lines: bool) -> None:
    conv.CLASS_MODE = class_mode
    conv.ACTIVE_LANE_CLASSES, conv.ACTIVE_LANE_CLASS_TO_ID = conv.build_lane_taxonomy(class_mode)
    conv.SKIP_LABELS = set() if include_stop_lines else {"stop_line"}


def discover_lane_folders(dataset_dir: Path) -> list[Path]:
    lane_folders: list[Path] = []
    for root, dirs, _files in os.walk(dataset_dir):
        dir_names = set(dirs)
        if {"images", "Annotations"}.issubset(dir_names):
            lane_folders.append(Path(root))
    lane_folders.sort()
    return lane_folders

def resolve_image_path(images_dir: Path, json_path: Path, payload: dict) -> Path | None:
    image_path_value = payload.get("imagePath")
    if isinstance(image_path_value, str) and image_path_value.strip():
        candidate = images_dir / Path(image_path_value).name
        if candidate.exists():
            return candidate

    stem = json_path.stem
    for ext in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def lane_folder_requires_flip(lane_folder: Path, dataset_dir: Path) -> bool:
    rel_parts = lane_folder.relative_to(dataset_dir).parts
    return any(part.endswith("_rot") for part in rel_parts)


def color_for_label(label: str, class_mode: int) -> tuple[int, int, int]:
    if class_mode == 3:
        return THREE_CLASS_COLOR_BGR.get(label, (180, 180, 180))
    return conv.bgr_color_for_lane_label(label)


def text_anchor(points: np.ndarray) -> tuple[int, int]:
    mid_idx = len(points) // 2
    x = int(points[mid_idx, 0])
    y = int(points[mid_idx, 1])
    return x, y


def draw_label_box(image: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    x0 = max(0, x - 2)
    y0 = max(0, y - text_h - baseline - 6)
    x1 = min(image.shape[1] - 1, x + text_w + 6)
    y1 = min(image.shape[0] - 1, y + 4)
    cv2.rectangle(image, (x0, y0), (x1, y1), (0, 0, 0), -1)
    cv2.putText(image, text, (x + 2, max(text_h + 2, y - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def build_shape_class_names(shapes: list[dict], img_width: int, class_mode: int) -> dict[int, str]:
    if class_mode == 3:
        return conv.assign_three_class_names(shapes, img_width=img_width)

    class_name_by_shape_idx: dict[int, str] = {}
    for idx, shape in enumerate(shapes):
        if not isinstance(shape, dict):
            continue
        label = str(shape.get("label", "")).strip()
        if label in conv.SKIP_LABELS:
            continue
        class_name_by_shape_idx[idx] = label
    return class_name_by_shape_idx


def render_annotation_preview(
    image: np.ndarray,
    shapes: list[dict],
    img_width: int,
    class_mode: int,
    line_width: int,
) -> np.ndarray:
    vis = image.copy()
    filtered_shapes = conv.filter_boundary_close_lanes_keep_curb(shapes, img_width=img_width)
    class_name_by_shape_idx = build_shape_class_names(filtered_shapes, img_width=img_width, class_mode=class_mode)

    for idx, shape in enumerate(filtered_shapes):
        if not isinstance(shape, dict):
            continue

        label = str(shape.get("label", "")).strip()
        if label in conv.SKIP_LABELS:
            continue

        pts = shape.get("points", [])
        if not isinstance(pts, list) or len(pts) < 2:
            continue

        class_label = class_name_by_shape_idx.get(idx, label if class_mode == 8 else "other_lanes")
        color = color_for_label(class_label, class_mode=class_mode)
        arr = np.asarray(pts, dtype=np.float32)
        thickness = max(2, conv.lane_thickness_for_points(arr, img_width=img_width, base=line_width))

        for i in range(len(arr) - 1):
            p0 = (int(arr[i, 0]), int(arr[i, 1]))
            p1 = (int(arr[i + 1, 0]), int(arr[i + 1, 1]))
            cv2.line(vis, p0, p1, color, thickness=thickness)

        for px, py in arr:
            center = (int(px), int(py))
            cv2.circle(vis, center, 5, (255, 255, 255), -1)
            cv2.circle(vis, center, 5, color, 2)

        tx, ty = text_anchor(arr.astype(np.int32))
        draw_label_box(vis, class_label, tx, ty, color)

    return vis


def analyze_shapes(
    shapes: list[dict],
    img_width: int,
    class_mode: int,
) -> tuple[int, Counter, bool]:
    filtered_shapes = conv.filter_boundary_close_lanes_keep_curb(shapes, img_width=img_width)
    class_name_by_shape_idx = build_shape_class_names(filtered_shapes, img_width=img_width, class_mode=class_mode)

    lane_count = 0
    class_counts: Counter = Counter()
    has_invisible_line = False

    for idx, shape in enumerate(filtered_shapes):
        if not isinstance(shape, dict):
            continue

        label = str(shape.get("label", "")).strip()
        if label in conv.SKIP_LABELS:
            continue

        pts = shape.get("points", [])
        if not isinstance(pts, list) or len(pts) < 2:
            continue

        if class_mode == 8:
            class_label = label
        else:
            class_label = class_name_by_shape_idx.get(idx, "other_lanes")
        class_counts[class_label] += 1
        if class_label == "invisible_line":
            has_invisible_line = True
        lane_count += 1

    return lane_count, class_counts, has_invisible_line


def output_path_for_image(output_root: Path, lane_folder: Path, dataset_dir: Path, image_path: Path) -> Path:
    del lane_folder, dataset_dir
    output_root.mkdir(parents=True, exist_ok=True)
    return output_root / image_path.name


def downsample_image(image: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return image
    height, width = image.shape[:2]
    new_width = max(1, width // factor)
    new_height = max(1, height // factor)
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


def write_rotated_source_image(
    rot_images_dir: Path,
    image_path: Path,
    image: np.ndarray,
    overwrite_rot_images: bool,
) -> Path:
    if overwrite_rot_images:
        cv2.imwrite(str(image_path), image)
        return image_path

    rot_images_dir.mkdir(parents=True, exist_ok=True)
    rotated_path = rot_images_dir / image_path.name
    cv2.imwrite(str(rotated_path), image)
    return rotated_path


def process_lane_folder(
    lane_folder: Path,
    dataset_dir: Path,
    class_mode: int,
    line_width: int,
    downsample_factor: int,
    overwrite_rot_images: bool = False,
    analyze_only: bool = False,
) -> tuple[int, int, int, Counter, Counter, list[str]]:
    images_dir = lane_folder / "images"
    labels_dir = lane_folder / "Annotations"
    output_root = lane_folder / "annotated_images"
    rot_images_dir = lane_folder / "rot_images"
    json_paths = sorted(labels_dir.glob("*.json"))
    should_flip_images = lane_folder_requires_flip(lane_folder, dataset_dir)

    seen = 0
    written = 0
    skipped = 0
    class_counts: Counter = Counter()
    lane_count_dist: Counter = Counter()
    invisible_line_images: list[str] = []

    for json_path in json_paths:
        seen += 1
        with json_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        shapes = payload.get("shapes", [])
        if not isinstance(shapes, list) or not shapes:
            skipped += 1
            continue

        img_width = int(payload.get("imageWidth", 0) or 0)
        if img_width <= 0:
            image_path = resolve_image_path(images_dir, json_path, payload)
            if image_path is None:
                skipped += 1
                continue
            image_for_width = cv2.imread(str(image_path))
            if image_for_width is None:
                skipped += 1
                continue
            img_width = int(image_for_width.shape[1])

        lane_count, image_class_counts, has_invisible_line = analyze_shapes(
            shapes=shapes,
            img_width=img_width,
            class_mode=class_mode,
        )
        class_counts.update(image_class_counts)
        lane_count_dist[lane_count] += 1
        # if has_invisible_line:
        #     invisible_line_images.append(str(json_path.relative_to(lane_folder)))

        if analyze_only:
            continue

        image_path = resolve_image_path(images_dir, json_path, payload)
        if image_path is None:
            skipped += 1
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            skipped += 1
            continue

        source_image_path = image_path
        if should_flip_images:
            image = cv2.flip(image, -1)
            source_image_path = write_rotated_source_image(
                rot_images_dir=rot_images_dir,
                image_path=image_path,
                image=image,
                overwrite_rot_images=overwrite_rot_images,
            )

        annotated = render_annotation_preview(
            image=image,
            shapes=shapes,
            img_width=img_width,
            class_mode=class_mode,
            line_width=line_width,
        )
        annotated_small = downsample_image(annotated, factor=downsample_factor)
        out_path = output_path_for_image(output_root, lane_folder, lane_folder, source_image_path)
        cv2.imwrite(str(out_path), annotated_small)
        written += 1

    return seen, written, skipped, class_counts, lane_count_dist, invisible_line_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create annotated_images previews directly from LabelMe lane annotations."
    )
    parser.add_argument(
        "--dataset-dir",
        required=True,
        type=Path,
        help="Root directory containing lane/session folders with images/ and labels/.",
    )
    parser.add_argument(
        "--class-mode",
        type=int,
        choices=[3, 8],
        default=8,
        help="Class mode: 8=original labelme taxonomy, 3=egoleft/egoright/other_lanes.",
    )
    parser.add_argument(
        "--include-stop-lines",
        action="store_true",
        default=False,
        help="Include stop_line annotations in the previews.",
    )
    parser.add_argument(
        "--line-width",
        type=int,
        default=6,
        help="Base line width used for rendering lane previews.",
    )
    parser.add_argument(
        "--downsample-factor",
        type=int,
        default=DEFAULT_DOWNSAMPLE_FACTOR,
        help="Downsample factor for annotated_images. Use 1 for no downsampling, 2 for half resolution, 4 for quarter resolution.",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        default=False,
        help="Only analyze label statistics. Do not write annotated_images or rot_images.",
    )
    parser.add_argument(
        "--overwrite-rot-images",
        action="store_true",
        default=False,
        help="When a lane folder requires rotation, overwrite the original image in images/ instead of writing corrected copies to rot_images/.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    if args.downsample_factor < 1:
        raise ValueError("--downsample-factor must be >= 1")

    configure_converter(class_mode=args.class_mode, include_stop_lines=args.include_stop_lines)

    lane_folders = discover_lane_folders(dataset_dir)
    if not lane_folders:
        raise FileNotFoundError(f"No lane folders with images/ and Annotations/ found under: {dataset_dir}")

    total_seen = 0
    total_written = 0
    total_skipped = 0
    total_class_counts: Counter = Counter()
    total_lane_count_dist: Counter = Counter()
    total_invisible_line_images: list[str] = []

    print(f"Dataset dir   : {dataset_dir}")
    print(f"Lane folders  : {len(lane_folders)}")
    print(f"Class mode    : {args.class_mode}")
    print(f"Stop lines    : {'included' if args.include_stop_lines else 'skipped'}")
    print(f"Downsample    : 1/{args.downsample_factor} in width and height")
    print(f"Analyze only  : {'yes' if args.analyze else 'no'}")
    if not args.analyze:
        if args.overwrite_rot_images:
            print("Rotate fix    : any discovered lane folder under a path ending with _rot will overwrite corrected images in images/")
        else:
            print("Rotate fix    : any discovered lane folder under a path ending with _rot will save corrected copies in rot_images/")

    for lane_folder in lane_folders:
        rel = lane_folder.relative_to(dataset_dir)
        seen, written, skipped, class_counts, lane_count_dist, invisible_line_images = process_lane_folder(
            lane_folder=lane_folder,
            dataset_dir=dataset_dir,
            class_mode=args.class_mode,
            line_width=args.line_width,
            downsample_factor=args.downsample_factor,
            overwrite_rot_images=args.overwrite_rot_images,
            analyze_only=args.analyze,
        )
        total_seen += seen
        total_written += written
        total_skipped += skipped
        total_class_counts.update(class_counts)
        total_lane_count_dist.update(lane_count_dist)
        total_invisible_line_images.extend(
            [f"{rel}/{path}" for path in invisible_line_images]
        )
        if not args.analyze:
            print(f"[{rel}] output: {lane_folder / 'annotated_images'}")
            if lane_folder_requires_flip(lane_folder, dataset_dir):
                if args.overwrite_rot_images:
                    print(f"[{rel}] rotated images overwritten in: {lane_folder / 'images'}")
                else:
                    print(f"[{rel}] rotated images: {lane_folder / 'rot_images'}")
        print(f"[{rel}] flip images: {'yes' if lane_folder_requires_flip(lane_folder, dataset_dir) else 'no'}")
        print(f"[{rel}] json: {seen}, written: {written}, skipped: {skipped}")
        if invisible_line_images:
            print(f"[{rel}] invisible_line images: {len(invisible_line_images)}")

    if args.analyze:
        print("\nAnalysis complete.")
    else:
        print("\nFinished creating annotated images.")
    print(f"  JSON files seen : {total_seen}")
    print(f"  Images written  : {total_written}")
    print(f"  Files skipped   : {total_skipped}")

    print("\n  Total class label counts:")
    for class_name in sorted(total_class_counts.keys()):
        print(f"    {class_name}: {total_class_counts[class_name]}")

    if total_invisible_line_images:
        print("\n  Images with invisible_line annotations:")
        for image_name in total_invisible_line_images:
            print(f"    {image_name}")

    print("\n  Lane count distribution per image:")
    for lane_count in range(1, 9):
        print(f"    {lane_count} lanes: {total_lane_count_dist.get(lane_count, 0)} images")
    if total_lane_count_dist.get(0, 0):
        print(f"    0 lanes: {total_lane_count_dist[0]} images")
    more_than_eight = sum(count for lanes, count in total_lane_count_dist.items() if lanes > 8)
    if more_than_eight:
        print(f"    >8 lanes: {more_than_eight} images")


if __name__ == "__main__":
    main()
