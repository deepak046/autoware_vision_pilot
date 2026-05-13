#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@dataclass(frozen=True)
class LaneFolder:
    """Dataset folder that contains LabelMe-style `images/` and `labels/` directories."""

    root: Path
    images_dir: Path
    labels_dir: Path
    relative_path: Path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for labelwise dataset extraction."""
    parser = argparse.ArgumentParser(
        description=(
            "Extract LabelMe JSON + image pairs into per-class folders for user-selected classes. "
            "Each class output contains only samples that include that class label."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Root directory containing one or more session folders with images/ and labels/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination directory where per-class extracted folders will be created.",
    )
    parser.add_argument(
        "--labels-dir-name",
        type=str,
        choices=["labels", "Annotations"],
        default="Annotations",
        help="Name of the source label directory to look for and recreate in the output.",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        required=True,
        help=(
            "Target class names to extract. Supports space-separated and/or comma-separated input. "
            "Example: --classes curb_line Occluded_curb_lines or --classes curb_line,Occluded_curb_lines"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Discover and report copy operations without writing files.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable debug-level logs.",
    )
    return parser.parse_args()


def configure_logging(verbose: bool) -> None:
    """Initialize application logger."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def normalize_class_list(raw_items: Iterable[str]) -> list[str]:
    """Normalize class list from CLI.

    Args:
        raw_items: Raw values from argparse `--classes`.

    Returns:
        Deduplicated class names preserving user order.

    Raises:
        ValueError: If no non-empty class names are provided.
    """
    ordered_classes: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        for token in item.split(","):
            class_name = token.strip()
            if not class_name or class_name in seen:
                continue
            seen.add(class_name)
            ordered_classes.append(class_name)
    if not ordered_classes:
        raise ValueError("No valid class names were provided in --classes.")
    return ordered_classes


def discover_lane_folders(dataset_dir: Path, labels_dir_name: str) -> list[LaneFolder]:
    """Find all subfolders that contain both `images/` and the requested labels directory."""
    lane_folders: list[LaneFolder] = []
    for root, dirs, _files in os.walk(dataset_dir):
        dir_names = set(dirs)
        if {"images", labels_dir_name}.issubset(dir_names):
            root_path = Path(root)
            lane_folders.append(
                LaneFolder(
                    root=root_path,
                    images_dir=root_path / "images",
                    labels_dir=root_path / labels_dir_name,
                    relative_path=root_path.relative_to(dataset_dir),
                )
            )
    lane_folders.sort(key=lambda item: str(item.relative_path))
    return lane_folders


def load_json_payload(json_path: Path) -> dict | None:
    """Load one LabelMe JSON payload.

    Args:
        json_path: JSON file path.

    Returns:
        Parsed dictionary when valid, otherwise `None`.
    """
    try:
        with json_path.open("r", encoding="utf-8") as file_obj:
            payload = json.load(file_obj)
    except (json.JSONDecodeError, OSError) as exc:
        logging.warning("Skipping invalid JSON %s (%s)", json_path, exc)
        return None
    if not isinstance(payload, dict):
        logging.warning("Skipping non-object JSON payload: %s", json_path)
        return None
    return payload


def labels_in_payload(payload: dict) -> set[str]:
    """Extract lane label names from LabelMe payload shapes."""
    labels: set[str] = set()
    shapes = payload.get("shapes", [])
    if not isinstance(shapes, list):
        return labels
    for shape in shapes:
        if not isinstance(shape, dict):
            continue
        label = str(shape.get("label", "")).strip()
        if label:
            labels.add(label)
    return labels


def resolve_image_path(images_dir: Path, json_path: Path, payload: dict) -> Path | None:
    """Resolve image corresponding to LabelMe JSON.

    Strategy:
    1) Honor payload `imagePath` basename if present.
    2) Fallback to same stem + known image extensions.
    """
    image_path_value = payload.get("imagePath")
    if isinstance(image_path_value, str) and image_path_value.strip():
        candidate = images_dir / Path(image_path_value).name
        if candidate.exists():
            return candidate

    stem = json_path.stem
    for extension in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{stem}{extension}"
        if candidate.exists():
            return candidate
    return None


def class_output_paths(
    output_dir: Path,
    class_name: str,
    lane_folder_relative: Path,
    labels_dir_name: str,
) -> tuple[Path, Path]:
    """Build destination images/labels directories for one class and one source session."""
    class_root = output_dir / class_name
    session_folder_name = flatten_session_name(lane_folder_relative)
    labels_target_dir = class_root / labels_dir_name / session_folder_name
    images_target_dir = class_root / "images" / session_folder_name
    return labels_target_dir, images_target_dir


def flatten_session_name(lane_folder_relative: Path) -> str:
    """Create a flat, unique session name by prepending parent path parts.

    Example:
      `city_a/day_01/session_0003` -> `city_a__day_01__session_0003`
    """
    parts = [part.strip() for part in lane_folder_relative.parts if part.strip()]
    if not parts:
        return "root_session"
    return "__".join(parts)


def copy_if_needed(src_path: Path, dst_path: Path, dry_run: bool) -> None:
    """Copy file while preserving metadata, unless dry-run mode is enabled."""
    if dry_run:
        return
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dst_path)


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    dataset_dir = args.dataset_dir.resolve()
    output_dir = args.output_dir.resolve()
    classes = normalize_class_list(args.classes)
    requested_classes: set[str] = set(classes)

    if not dataset_dir.exists() or not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset dir does not exist or is not a directory: {dataset_dir}")

    lane_folders = discover_lane_folders(dataset_dir, labels_dir_name=args.labels_dir_name)
    if not lane_folders:
        raise FileNotFoundError(
            f"No folders with images/ and {args.labels_dir_name}/ found under: {dataset_dir}"
        )

    logging.info("Dataset dir          : %s", dataset_dir)
    logging.info("Output dir           : %s", output_dir)
    logging.info("Target classes       : %s", ", ".join(classes))
    logging.info("Labels dir name      : %s", args.labels_dir_name)
    logging.info("Lane folders found   : %d", len(lane_folders))
    logging.info("Dry run              : %s", "yes" if args.dry_run else "no")

    total_json_seen = 0
    total_json_matched = 0
    total_images_missing = 0
    total_unpaired_skipped = 0
    total_json_invalid = 0
    per_class_json_count: Counter[str] = Counter()
    per_class_image_count: Counter[str] = Counter()
    class_to_samples: dict[str, list[str]] = defaultdict(list)

    for lane_folder in lane_folders:
        json_paths = sorted(lane_folder.labels_dir.glob("*.json"))
        if not json_paths:
            logging.debug("No JSON files in %s", lane_folder.labels_dir)
            continue

        for json_path in json_paths:
            total_json_seen += 1
            payload = load_json_payload(json_path)
            if payload is None:
                total_json_invalid += 1
                continue

            labels_present = labels_in_payload(payload)
            matched_classes = labels_present.intersection(requested_classes)
            if not matched_classes:
                continue

            total_json_matched += 1
            image_path = resolve_image_path(lane_folder.images_dir, json_path, payload)
            if image_path is None:
                total_images_missing += 1
                total_unpaired_skipped += 1
                logging.warning("Missing image for JSON: %s", json_path)
                continue

            for class_name in sorted(matched_classes):
                labels_target_dir, images_target_dir = class_output_paths(
                    output_dir=output_dir,
                    class_name=class_name,
                    lane_folder_relative=lane_folder.relative_path,
                    labels_dir_name=args.labels_dir_name,
                )
                target_json_path = labels_target_dir / json_path.name
                copy_if_needed(json_path, target_json_path, dry_run=args.dry_run)
                per_class_json_count[class_name] += 1

                target_image_path = images_target_dir / image_path.name
                copy_if_needed(image_path, target_image_path, dry_run=args.dry_run)
                per_class_image_count[class_name] += 1

                if len(class_to_samples[class_name]) < 5:
                    rel_sample = json_path.relative_to(dataset_dir)
                    class_to_samples[class_name].append(str(rel_sample))

    print("\nExtraction complete.")
    print(f"  JSON files seen      : {total_json_seen}")
    print(f"  JSON payload invalid : {total_json_invalid}")
    print(f"  JSON files matched   : {total_json_matched}")
    print(f"  Missing images       : {total_images_missing}")
    print(f"  Unpaired skipped     : {total_unpaired_skipped}")
    print(f"  Output root          : {output_dir}")
    print("\nPer-class copied samples:")
    for class_name in classes:
        print(
            f"  {class_name}: "
            f"json={per_class_json_count.get(class_name, 0)}, "
            f"images={per_class_image_count.get(class_name, 0)}"
        )
        for sample in class_to_samples.get(class_name, []):
            print(f"    - {sample}")


if __name__ == "__main__":
    main()
