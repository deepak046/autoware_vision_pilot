import cv2
import numpy as np
from torch.utils.data import Dataset

from Models.data_utils.lite_models.augmentation.factory import build_aug

# RGB colors per lane class (must match process_curvelanes.LANE_CLASS_SEMANTIC_RGB).
# Channel order: continuous_white_line, continuous_yellow_line, dashed_white_line,
# double_white_lines, double_yellow_lines, curb_line, stop_line, invisible_line.
LANE_CLASS_RGB_PALETTE_8 = [
    (240, 240, 240),
    (0, 255, 255),
    (200, 200, 200),
    (220, 220, 220),
    (0, 220, 255),
    (0, 140, 255),
    (0, 0, 255),
    (128, 128, 128),
]

LANE_CLASS_RGB_PALETTE_3 = [
    (0, 255, 255),
    (255, 0, 200),
    (0, 255, 145),
]


def rgb_semantic_mask_to_C_channels(rgb: np.ndarray, num_channels: int) -> np.ndarray:
    """
    HxWx3 uint8 RGB lane mask (background = anything not in the palette) ->
    HxWxC uint8 with values {0,1} per channel. Channel i is 1 where pixel equals
    LANE_CLASS_RGB_PALETTE[i] exactly.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(
            f"[BaseDataset] Expected RGB mask HxWx3, got shape {rgb.shape}"
        )
    h, w = rgb.shape[:2]
    out = np.zeros((h, w, num_channels), dtype=np.uint8)
    if num_channels == 3:
        palette = LANE_CLASS_RGB_PALETTE_3
    elif num_channels == 8:
        palette = LANE_CLASS_RGB_PALETTE_8
    elif num_channels == 1:
        palette = [(255, 0, 0)]
    else:
        raise ValueError(f"[BaseDataset] Expected 3 or 8 channels, got {num_channels}")

    for i, color in enumerate(palette):
        col = np.array(color, dtype=np.uint8)
        match = np.all(rgb == col, axis=-1)
        out[:, :, i] = match.astype(np.uint8)
    return out


class BaseDataset(Dataset):
    def __init__(self, dataset_root: str, aug_cfg: dict = {}, mode="train", data_type="SEGMENTATION", pseudo_labeling=False):
        """
        Pseudo labeling means that a larger model is used to generate labels for the unlabeled data (eventually used to generate depth maps from DepthAnythingV2 model).

        """
        self.aug_cfg = aug_cfg

        self.dataset_root = dataset_root

        self.mode = mode
            
        self.data_type = data_type

        self.pseudo_labeling = pseudo_labeling

        # ---- Build augmentations (does not know about pseudo-labeling) ----

        self.aug_type = self.data_type

        self.aug = build_aug(
            data_type=self.aug_type,
            cfg=aug_cfg,
            mode=self.mode,
            pseudo_labeling=self.pseudo_labeling
        )

        self.samples = []  # to be defined in child classes
        

    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        # TODO: Deepak
        img_path, gt_path = self.samples[idx]

        # --------------------------------------------------
        # 1) LOAD RAW IMAGE (BGR → RGB)
        # --------------------------------------------------
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # --------------------------------------------------
        # 2) LOAD / FAKE GT
        # --------------------------------------------------
        if self.pseudo_labeling is False:
            if self.data_type == "SEGMENTATION":
                label = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
            elif self.data_type == "DEPTH":
                label = np.load(gt_path)
            elif self.data_type == "LANE_DETECTION":
                label = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
                if label is None:
                    raise FileNotFoundError(f"[BaseDataset] Missing GT: {gt_path}")
                if label.ndim == 2:
                    raise ValueError(
                        f"[BaseDataset] Lane GT must be RGB (HxWx3), got HxW: {gt_path}"
                    )
                label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
            else:
                raise ValueError(
                    f"[BaseDataset] ERROR: unsupported data_type: {self.data_type}"
                )
        else:
            # fake GT (placeholder, will be ignored)
            if self.data_type == "SEGMENTATION":
                label = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
            elif self.data_type == "LANE_DETECTION":
                label = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
            elif self.data_type == "DEPTH":
                label = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)

        # --------------------------------------------------
        # 3) AUGMENTATION PIPELINE (IMAGE + GT)
        # --------------------------------------------------
        # TODO: Deepak | The augmentation normalizes the image but not the label
        image, label = self.aug.apply_augmentation(img, label, dataset_name=self.dataset_name)

        # --------------------------------------------------
        # 4) FINAL CAST FOR MODEL
        # --------------------------------------------------
        image = image.astype(np.float32)

        if self.data_type == "LANE_DETECTION":
            num_channels = self.aug_cfg.get("output_channels")
            if num_channels is not None:
                # RGB mask (HxWx3) -> C binary channels from exact palette matches
                label = rgb_semantic_mask_to_C_channels(label, num_channels) # Returns shape [H, W, C] with values {0,1} per class channel
                label = label.astype(np.float32)  # [H, W, C] with values {0.0, 1.0} per class channel
            else:
                # If it's only 3 binary channels
                label = label.astype(np.float32) / 255.0 # [H, W, 3] with values {0.0, 1.0} per class channel

            label = np.transpose(label, (2, 0, 1))  # [C,H,W] for BCE

            # Debug check for non-binary values
            u = np.unique(label)
            if len(u) > 2:
                print("WARN: non-binary mask values:", u[:20])
        
        else:
            # default behaviour for other tasks
            label = label.astype(np.int64)

        image = np.transpose(image, (2, 0, 1))  # CHW

        # --------------------------------------------------
        # 5) RETURN SAMPLE
        # --------------------------------------------------
        sample = {
            "image": image,
            "gt": label,
        }

        return sample



