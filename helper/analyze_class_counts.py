#!/usr/bin/env python3
"""
Analyze class label frequency from LabelMe-style annotation JSONs.

Expected dataset layout (typical for this repo):
  <root>/<session_name>/labels_json/*.json

Each JSON is expected to contain a `shapes` list where each element has:
  - `label`: class name
  - `points`: lane polyline points (ignored here)

The script outputs a dictionary:
  { "<session_name>": { "<class_label>": <count>, ... }, ... }
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple


def iter_labels_json_dirs(root: Path, labels_json_dir_name: str) -> List[Path]:
    """
    Return directories named `labels_json` under `root`.

    Strategy:
      1) Look for immediate children: <root>/<name>/labels_json
      2) If none found, fall back to a recursive scan for safety.
    """
    immediate = [p / labels_json_dir_name for p in root.iterdir() if p.is_dir()]
    found_immediate = [p for p in immediate if p.is_dir()]
    if found_immediate:
        return sorted(found_immediate)

    # Fallback: recurse; still bounded to directory name match only.
    return sorted([p for p in root.rglob(labels_json_dir_name) if p.is_dir()])


def extract_class_labels_from_annotation(payload: object) -> List[str]:
    """
    Extract `shape.label` strings from a LabelMe JSON payload.
    """
    if not isinstance(payload, dict):
        return []

    shapes = payload.get("shapes", [])
    if not isinstance(shapes, list):
        return []

    out: List[str] = []
    for shape in shapes:
        if not isinstance(shape, dict):
            continue
        label = shape.get("label", "")
        if label is None:
            continue
        label_str = str(label).strip()
        if label_str:
            out.append(label_str)
    return out


def count_labels_in_labels_json_dir(
    labels_json_dir: Path, json_glob: str, verbose: bool
) -> Tuple[Counter, Counter, int, List[str]]:
    """
    Returns:
      - label_counter: Counter(label -> total occurrences across all files in this dir)
      - files_by_lane_count: Counter(lane_count_per_file -> number_of_files_with_that_count)
      - max_lanes: maximum lane_count found for any file in this directory
      - files_with_max_lanes: list of JSON filenames that have lane_count == max_lanes
    """
    counter: Counter = Counter()
    files_by_lane_count: Counter = Counter()
    json_files = sorted(labels_json_dir.glob(json_glob))
    if verbose:
        print(f"[INFO] Scanning {len(json_files)} files in: {labels_json_dir}")

    max_lanes = -1
    files_with_max_lanes: List[str] = []

    for jf in json_files:
        try:
            payload = json.loads(jf.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            if verbose:
                print(f"[WARN] Failed to read/parse: {jf}")
            continue

        labels = extract_class_labels_from_annotation(payload)
        # "Lane count" here means: number of shape entries that have a non-empty label
        # (i.e., the same thing being counted into `labels`).
        lane_count = len(labels)
        files_by_lane_count[lane_count] += 1
        if lane_count > max_lanes:
            max_lanes = lane_count
            files_with_max_lanes = [jf.name]
        elif lane_count == max_lanes:
            files_with_max_lanes.append(jf.name)
        # Count each lane/class occurrence (per shape) within this image.
        counter.update(labels)

    return counter, files_by_lane_count, max_lanes, files_with_max_lanes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count class label occurrences inside LabelMe JSON files under labels_json directories."
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Root directory containing session folders (each with a labels_json/ folder).",
    )
    parser.add_argument(
        "--labels-json-dir-name",
        type=str,
        default="labels_json",
        help="Name of the directory that contains label JSON files (default: labels_json).",
    )
    parser.add_argument(
        "--json-glob",
        type=str,
        default="*.json",
        help="Glob pattern for JSON files inside labels_json directories (default: *.json).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress information.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root: Path = args.root.expanduser().resolve()

    labels_json_dirs = iter_labels_json_dirs(root, args.labels_json_dir_name)

    # dataset_name -> Counter(label -> count)
    per_dataset: Dict[str, Counter] = {}
    per_dataset_files_by_lane_count: Dict[str, Counter] = {}
    per_dataset_max_lanes_files: Dict[str, Dict[str, object]] = {}
    global_counter: Counter = Counter()
    global_files_by_lane_count: Counter = Counter()

    for ljd in labels_json_dirs:
        dataset_name = ljd.parent.name  # <root>/<dataset_name>/labels_json
        print(f"[Processing dataset: {dataset_name}]")
        counter, files_by_lane_count, max_lanes, files_with_max_lanes = (
            count_labels_in_labels_json_dir(ljd, args.json_glob, verbose=args.verbose)
        )
        per_dataset[dataset_name] = counter
        per_dataset_files_by_lane_count[dataset_name] = files_by_lane_count
        global_counter.update(counter)
        global_files_by_lane_count.update(files_by_lane_count)
        per_dataset_max_lanes_files[dataset_name] = {
            "max_lanes": int(max_lanes),
            "files": sorted(files_with_max_lanes),
        }
        if files_with_max_lanes:
            print(
                f"[Max lanes] {dataset_name}: max_lanes={max_lanes}, num_files={len(files_with_max_lanes)}"
            )

    # Convert Counters to plain dicts for stable JSON output.
    dataset_dict: Dict[str, Dict[str, int]] = {
        ds: dict(counter) for ds, counter in sorted(per_dataset.items())
    }
    dataset_files_by_lane_count_dict: Dict[str, Dict[int, int]] = {
        ds: {int(k): int(v) for k, v in sorted(counter.items())}
        for ds, counter in sorted(per_dataset_files_by_lane_count.items())
    }

    result = {
        "datasets": dataset_dict,
        "files_by_lane_count_per_dataset": dataset_files_by_lane_count_dict,
        "global_class_counts": dict(global_counter),
        "global_files_by_lane_count": {int(k): int(v) for k, v in sorted(global_files_by_lane_count.items())},
        "files_with_max_lanes_per_dataset": per_dataset_max_lanes_files,
        "num_datasets": len(dataset_dict),
        "total_unique_labels": len(global_counter),
    }

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
