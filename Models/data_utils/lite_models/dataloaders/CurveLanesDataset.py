# dataloader/CurveLanesDataset.py

import os
import glob
import json
from collections import Counter
from typing import Dict, Iterable, List, Optional, Set, Tuple

import cv2
import numpy as np

from Models.data_utils.lite_models.dataloaders.BaseDataset import BaseDataset


"""
CurveLanes Dataset (processed version).

Expected structure:

CurveLanes/processed/
├── image/            (*.jpg)
├── mask/             (*.png)
├── visualization/
└── drivable_path.json

Each image must have a corresponding mask with the same basename.
Example:
    image/000123.jpg
    mask/000123.png
"""

LANE_CLASSES_9 = [
    "continuous_white_line",
    "continuous_yellow_line",
    "dashed_white_line",
    "double_white_lines",
    "double_yellow_lines",
    "curb_line",
    "stop_line",
    "invisible_line",
    "Occluded_curb_lines",
]
LANE_CLASS_TO_ID_9 = {name: idx for idx, name in enumerate(LANE_CLASSES_9)}
LANE_RGB_PALETTE_9 = [
    (240, 240, 240),
    (0, 255, 255),
    (200, 200, 200),
    (220, 220, 220),
    (0, 220, 255),
    (0, 140, 255),
    (0, 0, 255),
    (128, 128, 128),
    (255, 0, 255),
]

class CurveLanesDataset(BaseDataset):

    def __init__(
        self,
        dataset_root: str,
        aug_cfg: dict = {},
        mode: str = "train",
        data_type: str = "LANE_DETECTION",
        pseudo_labeling: bool = False,
        annotated_lane_classes: Optional[Dict[str, List[str]]] = None,
    ):
        super().__init__(
            dataset_root,
            aug_cfg=aug_cfg,
            mode=mode,
            data_type=data_type,
            pseudo_labeling=pseudo_labeling,
        )

        self.root = dataset_root

        # Force processed version
        if "processed" not in os.path.basename(self.root):
            print(
                "[CurveLanesDataset] WARNING: dataset_root does not point to 'processed/'. "
                "Appending '/processed'."
            )
            self.root = os.path.join(self.root, "processed")

        self.split = mode.lower()   # "train" | "val" | "test"
        self.dataset_name = "Curvelanes"
        self.annotated_lane_classes: Dict[str, List[str]] = {}

        if self.data_type != "LANE_DETECTION":
            raise ValueError(
                f"[CurveLanesDataset] Unsupported data_type: {self.data_type}. "
                "Only 'LANE_DETECTION' is supported."
            )

        # Dict of annotated lane classes within each image
        if annotated_lane_classes is not None:
            self.annotated_lane_classes = annotated_lane_classes

        # ---- Build file list ----
        self.samples = self._build_file_list()

    # ------------------------------------------------------------
    # Build file list 
    # ------------------------------------------------------------
    def _build_file_list(self) -> List[Tuple[str, str]]:

        MAX_VAL_SAMPLES = 2000       #limit max number of validation samples to 500 (otherwise they would be )

        """
        Build file list for CurveLanes dataset.:
            - Sort all frames deterministically
            - Every 10th sample goes to validation
            - The rest goes to training

        Returns:
            samples: list[(img_path, gt_path)]
        """

        print(
            f"[CurveLanesDataset] Building file list for split='{self.split}', "
            f"data_type='{self.data_type}'"
        )

        image_root = os.path.join(self.root, "CURVELANES", "image")
        mask_root  = os.path.join(self.root, "CURVELANES", "mask")

        if not os.path.isdir(image_root):
            raise FileNotFoundError(f"Missing image directory: {image_root}")
        if not os.path.isdir(mask_root):
            raise FileNotFoundError(f"Missing mask directory: {mask_root}")

        # --------------------------------------------------
        # Collect and sort all images
        # --------------------------------------------------
        img_files = sorted(
            glob.glob(os.path.join(image_root, "*.jpg"))
        )

        if len(img_files) == 0:
            raise RuntimeError(f"[CurveLanesDataset] No images found in {image_root}")

        print(f"[CurveLanesDataset] Found {len(img_files)} images total.")

        samples = []

        # --------------------------------------------------
        # Deterministic split
        # --------------------------------------------------
        for idx, img_path in enumerate(img_files):

            basename = os.path.splitext(os.path.basename(img_path))[0]
            gt_path = os.path.join(mask_root, f"{basename}.png")

            if not os.path.isfile(gt_path):
                print(f"[CurveLanesDataset] WARNING: Missing GT mask for {img_path}")
                continue

            is_val = (idx % 10 == 0)

            if self.split == "train" and not is_val:
                samples.append((img_path, gt_path))

            elif self.split == "val" and is_val and (len(samples) < MAX_VAL_SAMPLES):
                #cap validation samples to MAX_VAL_SAMPLES
                samples.append((img_path, gt_path))
            


        print(
            f"[CurveLanesDataset] Loaded {len(samples)} samples for split='{self.split}'."
        )

        if len(samples) == 0:
            raise RuntimeError(
                f"[CurveLanesDataset] Empty dataset split='{self.split}'. "
                "Check dataset path and split logic."
            )

        return samples

    def _present_class_ids_from_mask(self, mask_path: str) -> Set[int]:
        """Extract present class ids from precomputed per-image class metadata."""
        basename = os.path.splitext(os.path.basename(mask_path))[0]
        candidate_keys = (f"{basename}.jpg", f"{basename}.png", basename)
        class_names: Optional[List[str]] = None
        for key in candidate_keys:
            if key in self.annotated_lane_classes:
                class_names = self.annotated_lane_classes[key]
                break

        if class_names is None:
            return set()

        present_ids: Set[int] = set()
        for class_name in class_names:
            if class_name in LANE_CLASS_TO_ID_9:
                present_ids.add(LANE_CLASS_TO_ID_9[class_name])
            else:
                print(
                    f"[CurveLanesDataset] WARNING: Unknown lane class '{class_name}' "
                    f"in annotated_lane_classes for '{basename}'."
                )
        return present_ids

    def build_sample_weights(
        self,
        class_weights: Dict[str, float],
        min_weight: float = 1.0,
    ) -> List[float]:
        """Build per-sample weights for `WeightedRandomSampler`.

        Weight rule (requested):
            sample_weight = max(weight[c] for c in classes_present_in_image)
        No multiplicative stacking is used when multiple rare classes are present.

        Args:
            class_weights: Class->weight dictionary. Keys should follow `LANE_CLASSES_9`.
            min_weight: Default/floor weight for images where no configured class is found.

        Returns:
            List of weights aligned with `self.samples` order.
        """
        if min_weight <= 0:
            raise ValueError(f"[CurveLanesDataset] min_weight must be > 0, got {min_weight}")

        raw_weights: Dict[int, float] = {}
        for class_name, weight in class_weights.items():
            if class_name not in LANE_CLASS_TO_ID_9:
                print(
                    f"[CurveLanesDataset] WARNING: Unknown class '{class_name}' in class_weights; skipping."
                )
                continue
            weight_value = float(weight)
            if weight_value <= 0:
                print(
                    f"[CurveLanesDataset] WARNING: Non-positive weight for '{class_name}' ({weight_value}); skipping."
                )
                continue
            raw_weights[LANE_CLASS_TO_ID_9[class_name]] = weight_value

        if not raw_weights:
            raise ValueError(
                "[CurveLanesDataset] No valid class weights provided. "
                "Expected keys from 9-class taxonomy with positive values."
            )

        weight_sum = float(sum(raw_weights.values()))
        if weight_sum <= 0:
            raise ValueError(
                f"[CurveLanesDataset] Invalid weight sum ({weight_sum}). "
                "All class weights must be positive."
            )
        normalized_weights = {
            class_id: weight_value / weight_sum for class_id, weight_value in raw_weights.items()
        }
        normalized_min_weight = float(min_weight) / weight_sum

        sample_weights: List[float] = []
        per_class_presence: Counter = Counter()
        for _img_path, gt_path in self.samples:
            present_ids = self._present_class_ids_from_mask(gt_path)
            candidate_weights = [
                normalized_weights[class_id]
                for class_id in present_ids
                if class_id in normalized_weights
            ]
            sample_weight = max(candidate_weights) if candidate_weights else normalized_min_weight
            sample_weights.append(sample_weight)

            for class_id in present_ids:
                per_class_presence[LANE_CLASSES_9[class_id]] += 1

        print(
            f"[CurveLanesDataset] Weighted sampling metadata ready: {len(sample_weights)} samples."
        )
        print(
            f"[CurveLanesDataset] Present-class coverage: "
            + ", ".join(
                f"{class_name}={per_class_presence.get(class_name, 0)}"
                for class_name in LANE_CLASSES_9
            )
        )
        return sample_weights
