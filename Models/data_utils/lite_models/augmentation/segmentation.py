# dataloader/augmentations/segmentation.py
import albumentations as A
from Models.data_utils.lite_models.augmentation.BaseAugmentation import BaseAugmentation
import random
import cv2

import os
import numpy as np

import torch


class SegmentationAugmentation(BaseAugmentation):
    def _build_color_transforms(self):
        """Build image-only color-space transforms from cfg.color_aug."""
        color_cfg = self.cfg.get("color_aug", {}) or {}
        if not bool(color_cfg.get("enabled", False)):
            return []

        tfs = []

        # brightness, contrast, saturation, hue
        cj = color_cfg.get("color_jitter", [0.2, 0.2, 0.2, 0.1])
        if isinstance(cj, (list, tuple)) and len(cj) == 4:
            tfs.append(
                A.ColorJitter(
                    brightness=float(cj[0]),
                    contrast=float(cj[1]),
                    saturation=float(cj[2]),
                    hue=float(cj[3]),
                    p=float(color_cfg.get("p_color_jitter", 0.0)),
                )
            )

        # hue/saturation/value shifts in OpenCV range
        hsv = color_cfg.get("hsv_shift_limit", [10, 20, 20])
        if isinstance(hsv, (list, tuple)) and len(hsv) == 3:
            tfs.append(
                A.HueSaturationValue(
                    hue_shift_limit=int(hsv[0]),
                    sat_shift_limit=int(hsv[1]),
                    val_shift_limit=int(hsv[2]),
                    p=float(color_cfg.get("p_hsv", 0.0)),
                )
            )

        gamma_limit = color_cfg.get("random_gamma_limit", [80, 120])
        if isinstance(gamma_limit, (list, tuple)) and len(gamma_limit) == 2:
            tfs.append(
                A.RandomGamma(
                    gamma_limit=(int(gamma_limit[0]), int(gamma_limit[1])),
                    p=float(color_cfg.get("p_gamma", 0.0)),
                )
            )

        clahe_limit = color_cfg.get("clahe_clip_limit", [1.0, 4.0])
        clahe_grid = color_cfg.get("clahe_tile_grid_size", [8, 8])
        if (
            isinstance(clahe_limit, (list, tuple))
            and len(clahe_limit) == 2
            and isinstance(clahe_grid, (list, tuple))
            and len(clahe_grid) == 2
        ):
            tfs.append(
                A.CLAHE(
                    clip_limit=(float(clahe_limit[0]), float(clahe_limit[1])),
                    tile_grid_size=(int(clahe_grid[0]), int(clahe_grid[1])),
                    p=float(color_cfg.get("p_clahe", 0.0)),
                )
            )

        tfs.append(
            A.ToGray(
                num_output_channels=3,
                p=float(color_cfg.get("to_gray_prob", 0.0)),
            )
        )
        tfs.append(
            A.ChannelShuffle(
                p=float(color_cfg.get("p_channel_shuffle", 0.0)),
            )
        )

        return tfs

    def _build(self):
        """
        GEOMETRY ONLY.
        (Noise + Normalize are applied in apply_augmentation via BaseAugmentation._postprocess_image)
          rescaling:
            enabled: true
            mode: "fixed_resize"    # fixed_resize | random_crop


            width: 1024       
            height: 512

            #for random cropping
            scale_range: [0.5, 2.0]   # min and max scale for random cropping

        """
        tfs = []
        mode = self.cfg.get("rescaling", {}).get("mode", "fixed_resize")


        #Albumentations Resize interpolation. Default is cv2.INTER_LINEAR for images and cv2.INTER_NEAREST for masks (so default is ok for segmentation)
        if self.mode == "train":
            if mode == "fixed_resize":
                tfs.append(
                    A.Resize(
                        height=int(self.cfg["rescaling"]["height"]),
                        width=int(self.cfg["rescaling"]["width"]),
                    )
                )

            elif mode == "random_crop":
                crop_h = int(self.cfg["rescaling"]["height"])
                crop_w = int(self.cfg["rescaling"]["width"])

                scale_min, scale_max = self.cfg["rescaling"]["scale_range"]

                # shared state per image+mask (per-sample)
                shared = {}

                def sample_valid_scale(h, w):
                    for _ in range(10):
                        scale = random.randint(
                            int(scale_min * 10),
                            int(scale_max * 10)
                        ) / 10.0
                        if h * scale >= crop_h and w * scale >= crop_w:
                            return scale
                    return max(crop_h / h, crop_w / w)

                def scale_image(img, **kwargs):
                    h, w = img.shape[:2]
                    scale = sample_valid_scale(h, w)
                    shared["scale"] = scale  
                    return cv2.resize(
                        img,
                        (int(w * scale), int(h * scale)),
                        interpolation=cv2.INTER_LINEAR,
                    )

                def scale_mask(mask, **kwargs):
                    h, w = mask.shape[:2]
                    scale = shared["scale"] 
                    return cv2.resize(
                        mask,
                        (int(w * scale), int(h * scale)),
                        interpolation=cv2.INTER_NEAREST,
                    )

                tfs.extend([
                    # 1) global multi-scale
                    A.Lambda(
                        image=scale_image,
                        mask=scale_mask,
                    ),
                    # 2) min padding
                    A.PadIfNeeded(
                        min_height=crop_h,
                        min_width=crop_w,
                        border_mode=0,
                        value=0,
                        mask_value=255,
                    ),
                    # 3) random crop to final size
                    A.RandomCrop(height=crop_h, width=crop_w),
                ])


            else:
                raise ValueError(f"Unknown segmentation mode: {mode}")

            flip_p = float(self.cfg.get("flip_prob", 0.0))
            if flip_p > 0:
                tfs.append(A.HorizontalFlip(p=flip_p))

            # image-only color space transforms
            tfs.extend(self._build_color_transforms())

        else:
            # validation / test
            if mode == "fixed_resize":
                # explicit fixed-res validation 
                tfs.append(
                    A.Resize(
                        height=int(self.cfg["rescaling"]["height"]),
                        width=int(self.cfg["rescaling"]["width"]),
                    )
                )

            else:
                # NO crop, NO resize
                # only pad to make H,W divisible by output stride (e.g. 16)
                output_stride = 16      #used by deeplabv3plus

                tfs.append(
                    A.PadIfNeeded(
                        min_height=None,
                        min_width=None,
                        pad_height_divisor=output_stride,
                        pad_width_divisor=output_stride,
                        border_mode=cv2.BORDER_CONSTANT,
                        value=0,          # image padding
                        mask_value=255,   # IGNORE label
                    )
                )

        self.transform = A.Compose(tfs)

    def apply_augmentation(self, image, label, dataset_name=None):
        # 1) remove car hood
        image, label = self._remove_car_hood(image, label, dataset_name)

        # 2) geometry (image+label)
        out = self.transform(image=image, mask=label)
        image, label = out["image"], out["mask"]


        # 3) noise + normalize (image only)
        image = self._apply_noise_img(image)
        image = self._apply_normalize(image)

        return image, label
