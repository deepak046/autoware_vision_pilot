#!/usr/bin/env python3

from __future__ import annotations

import argparse
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@dataclass(frozen=True)
class SessionFolder:
    """A nested session folder that contains both images and annotations."""

    root: Path
    images_dir: Path
    annotations_dir: Path
    relative_path: Path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Copy nested session folders while filtering images. "
            "Only image files with matching JSON in Annotations are copied."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Root directory containing nested session folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output root where filtered session structure is copied.",
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
        help="Enable verbose logging.",
    )
    return parser.parse_args()


def configure_logging(verbose: bool) -> None:
    """Configure global logger."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def discover_sessions(input_dir: Path) -> list[SessionFolder]:
    """Discover nested session folders that contain annotations and image sources."""
    sessions: list[SessionFolder] = []
    for root, dirs, _files in os.walk(input_dir):
        dir_names = set(dirs)
        has_annotations = "Annotations" in dir_names
        has_any_images = "images" in dir_names or "rot_images" in dir_names
        if has_annotations and has_any_images:
            root_path = Path(root)
            sessions.append(
                SessionFolder(
                    root=root_path,
                    images_dir=root_path / "images",
                    annotations_dir=root_path / "Annotations",
                    relative_path=root_path.relative_to(input_dir),
                )
            )
    sessions.sort(key=lambda item: str(item.relative_path))
    return sessions


def copy_if_needed(src: Path, dst: Path, dry_run: bool) -> None:
    """Copy one file while preserving metadata."""
    if dry_run:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def is_rot_session(session: SessionFolder) -> bool:
    """Return True when any folder segment ends with '_rot'."""
    return any(path_part.endswith("_rot") for path_part in session.relative_path.parts)


def resolve_source_images_dir(session: SessionFolder) -> Path:
    """Pick source image directory by session naming convention."""
    if is_rot_session(session):
        return session.root / "rot_images"
    return session.images_dir


def process_session(session: SessionFolder, input_dir: Path, output_dir: Path, dry_run: bool) -> tuple[int, int]:
    """Copy one session with filtered images.

    Returns:
        Tuple of (images_copied, images_skipped).
    """
    del input_dir  # kept for explicit signature symmetry
    output_session_root = output_dir / session.relative_path
    output_images_dir = output_session_root / "images"
    output_annotations_dir = output_session_root / "labels" # while saving output we will name it "labels" instead of "Annotations"

    copied = 0
    skipped = 0
    source_images_dir = resolve_source_images_dir(session)

    json_paths = sorted(session.annotations_dir.glob("*.json"))
    valid_stems = {json_path.stem for json_path in json_paths}

    if not dry_run:
        output_annotations_dir.mkdir(parents=True, exist_ok=True)
    for json_path in json_paths:
        target_json = output_annotations_dir / json_path.name
        copy_if_needed(json_path, target_json, dry_run=dry_run)

    if not source_images_dir.exists() or not source_images_dir.is_dir():
        logging.warning("[%s] missing source image dir: %s", session.relative_path, source_images_dir)
        return copied, len(valid_stems)

    image_paths: list[Path] = []
    for ext in IMAGE_EXTENSIONS:
        image_paths.extend(sorted(source_images_dir.glob(f"*{ext}")))
        image_paths.extend(sorted(source_images_dir.glob(f"*{ext.upper()}")))

    seen_image_names: set[str] = set()
    for image_path in image_paths:
        if image_path.name in seen_image_names:
            continue
        seen_image_names.add(image_path.name)

        if image_path.stem not in valid_stems:
            skipped += 1
            continue

        target_image = output_images_dir / image_path.name
        copy_if_needed(image_path, target_image, dry_run=dry_run)
        copied += 1

    return copied, skipped


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist or is not a directory: {input_dir}")

    sessions = discover_sessions(input_dir)
    if not sessions:
        raise FileNotFoundError(f"No session folders with images/ and Annotations/ found under: {input_dir}")

    logging.info("Input root      : %s", input_dir)
    logging.info("Output root     : %s", output_dir)
    logging.info("Sessions found  : %d", len(sessions))
    logging.info("Dry run         : %s", "yes" if args.dry_run else "no")

    total_images_copied = 0
    total_images_skipped = 0
    total_annotations_copied = 0

    for session in sessions:
        json_count = len(list(session.annotations_dir.glob("*.json")))
        copied, skipped = process_session(
            session=session,
            input_dir=input_dir,
            output_dir=output_dir,
            dry_run=args.dry_run,
        )
        total_images_copied += copied
        total_images_skipped += skipped
        total_annotations_copied += json_count
        logging.info(
            "[%s] copied images=%d, skipped images=%d, copied annotations=%d",
            session.relative_path,
            copied,
            skipped,
            json_count,
        )

    print("\nFiltering copy complete.")
    print(f"  Sessions processed     : {len(sessions)}")
    print(f"  Images copied          : {total_images_copied}")
    print(f"  Images skipped         : {total_images_skipped}")
    print(f"  Annotations copied     : {total_annotations_copied}")
    print(f"  Output root            : {output_dir}")


if __name__ == "__main__":
    main()
