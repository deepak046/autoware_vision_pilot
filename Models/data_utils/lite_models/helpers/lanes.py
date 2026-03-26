# utils/utils_lanes.py

import numpy as np
import cv2
import torch
from tqdm import tqdm
from typing import Dict

import torch.nn.functional as F


# ============================================================
# Visualization utils
# ============================================================

LANE_COLORS_RGB = {
    "egoleft":  (0, 255, 255),   # cyan
    "egoright": (255, 0, 200),   # magenta-ish
    "other":    (0, 255, 145),   # green
}

# 8-class lane taxonomy RGB (must match BaseDataset.LANE_CLASS_RGB_PALETTE / process_curvelanes).
LANE_CLASS_NAMES_8 = [
    "continuous_white_line",
    "continuous_yellow_line",
    "dashed_white_line",
    "double_white_lines",
    "double_yellow_lines",
    "curb_line",
    "stop_line",
    "invisible_line",
]
LANE_CLASS_COLORS_RGB_8 = [
    (240, 240, 240),
    (0, 255, 255),
    (200, 200, 200),
    (220, 220, 220),
    (0, 220, 255),
    (0, 140, 255),
    (0, 0, 255),
    (128, 128, 128),
]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def denorm_image_chw_to_uint8(image_tensor: torch.Tensor) -> np.ndarray:
    img = image_tensor.detach().cpu().float().numpy()  # CHW
    img = img.transpose(1, 2, 0)  # HWC
    img = (img * IMAGENET_STD + IMAGENET_MEAN) * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def _apply_lane_colors_rgb(canvas: np.ndarray, mask3: np.ndarray) -> np.ndarray:
    out = canvas.copy()

    if mask3.dtype != np.bool_:
        if mask3.max() > 1.5:
            mask3 = (mask3 > 127)
        else:
            mask3 = (mask3 > 0.5)

    # left
    ys, xs = np.where(mask3[..., 0])
    out[ys, xs, :] = LANE_COLORS_RGB["egoleft"]

    # right
    ys, xs = np.where(mask3[..., 1])
    out[ys, xs, :] = LANE_COLORS_RGB["egoright"]

    # other
    ys, xs = np.where(mask3[..., 2])
    out[ys, xs, :] = LANE_COLORS_RGB["other"]

    return out

def logits_to_lane_maskC(
    logits_chw: torch.Tensor,
    threshold: float = 0.5,
    use_sigmoid: bool = True,
) -> np.ndarray:
    """
    logits_chw: [C, H, W] — one logit map per lane class.
    Returns HxWxC bool (independent channels; overlaps possible).
    """
    x = logits_chw.detach()
    if use_sigmoid:
        x = torch.sigmoid(x)
        mask = (x > threshold)
    else:
        mask = (x > threshold)
    return mask.permute(1, 2, 0).cpu().numpy().astype(bool)


def logits_to_lane_maskC_argmax(
    logits_chw: torch.Tensor,
    use_sigmoid: bool = True,
) -> np.ndarray:
    """
    Single winning class per pixel (mutually exclusive one-hot HxWx8).
    logits_chw: [C, H, W]
    """
    x = logits_chw.detach()
    C = int(logits_chw.shape[0])
    if use_sigmoid:
        x = torch.sigmoid(x)
    winner = x.argmax(dim=0)
    mask = torch.zeros_like(x, dtype=torch.bool)
    for c in range(C):
        mask[c] = winner == c
    return mask.permute(1, 2, 0).cpu().numpy()


def logits_to_lane_class_id_hw(
    logits_chw: torch.Tensor,
    use_sigmoid: bool = True,
) -> np.ndarray:
    """
    Per-pixel class index in {0..7} from argmax over channels.
    Shape [H, W], dtype int64.
    """
    x = logits_chw.detach()
    if use_sigmoid:
        x = torch.sigmoid(x)
    return x.argmax(dim=0).cpu().numpy().astype(np.int64)


def apply_lane_colors_rgb_8class(canvas: np.ndarray, mask8: np.ndarray) -> np.ndarray:
    """
    canvas: HxWx3 uint8 RGB
    mask8: HxWx8 bool — multi-label or one-hot; paints class colors (later indices
    overwrite earlier on overlaps).
    """
    out = canvas.copy()
    if mask8.dtype != np.bool_:
        if mask8.max() > 1.5:
            mask8 = mask8 > 127
        else:
            mask8 = mask8 > 0.5
    for i, color in enumerate(LANE_CLASS_COLORS_RGB_8):
        ys, xs = np.where(mask8[..., i])
        out[ys, xs, :] = color
    return out


def tensor_maskC_to_numpy(maskC_chw: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
    """
    GT: [C,H,W] -> HxWxC bool
    """
    m = maskC_chw.detach().cpu().float()
    mask = (m > threshold)
    return mask.permute(1, 2, 0).numpy().astype(bool)


def make_lane_vis_pair_C_class(
    image_chw: torch.Tensor,
    pred_logits_chw: torch.Tensor,
    gt_maskC_chw: torch.Tensor,
    *,
    alpha: float = 0.5,
    pred_threshold: float = 0.0,
    pred_use_sigmoid: bool = False,
    pred_use_argmax: bool = True,
):
    """
    Generic 2x2 visualization for multi-class lane heads.

    For C==3: falls back to the existing binary-per-channel (egoleft/egoright/other) visualization.
    For C==8: uses argmax over channels for a mutually-exclusive mask to avoid
    painting the whole frame with one class.
    """
    base_img = denorm_image_chw_to_uint8(image_chw)
    H_img, W_img, _ = base_img.shape
    C = int(gt_maskC_chw.shape[0])
    Hm, Wm = int(gt_maskC_chw.shape[1]), int(gt_maskC_chw.shape[2])

    if (H_img != Hm) or (W_img != Wm):
        base_img_small = cv2.resize(base_img, (Wm, Hm), interpolation=cv2.INTER_LINEAR)
    else:
        base_img_small = base_img

    gt_maskC = tensor_maskC_to_numpy(gt_maskC_chw, threshold=0.5)

    if C == 3:
        # logits_to_lane_mask3 expects [3,H,W] and returns bool HxWx3.
        pred_mask3 = logits_to_lane_maskC(
            pred_logits_chw, threshold=pred_threshold, use_sigmoid=pred_use_sigmoid
        )
        gt_mask3 = gt_maskC  # HxWx3
        pred_colored = _apply_lane_colors_rgb(base_img_small, pred_mask3)
        gt_colored = _apply_lane_colors_rgb(base_img_small, gt_mask3)
    elif C == 8:
        # TODO: Deepak
        # if pred_use_argmax:
        #     pred_mask8 = logits_to_lane_maskC_argmax(
        #         pred_logits_chw, use_sigmoid=pred_use_sigmoid
        #     )
        # else:
        pred_mask8 = logits_to_lane_maskC(
            pred_logits_chw,
            threshold=pred_threshold,
            use_sigmoid=pred_use_sigmoid,
        )
        pred_colored = apply_lane_colors_rgb_8class(base_img_small, pred_mask8)
        gt_colored = apply_lane_colors_rgb_8class(base_img_small, gt_maskC)
    else:
        # Unknown channel count: skip visualization.
        return np.zeros((Hm * 2, Wm * 2, 3), dtype=np.uint8)

    pred_normal = cv2.addWeighted(pred_colored, alpha, base_img_small, 1 - alpha, 0)
    gt_normal = cv2.addWeighted(gt_colored, alpha, base_img_small, 1 - alpha, 0)

    black = np.zeros((Hm, Wm, 3), dtype=np.uint8)
    if C == 3:
        pred_raw = _apply_lane_colors_rgb(black, pred_mask3)
        gt_raw = _apply_lane_colors_rgb(black, gt_mask3)
    else:
        pred_raw = apply_lane_colors_rgb_8class(black, pred_mask8)
        gt_raw = apply_lane_colors_rgb_8class(black, gt_maskC)

    top = np.concatenate([pred_normal, gt_normal], axis=1)
    bot = np.concatenate([pred_raw, gt_raw], axis=1)
    tile = np.concatenate([top, bot], axis=0)
    return tile


# ============================================================
# Metrics utils (binary per-channel)
# ============================================================

def update_binary_confmat(confmat, pred: np.ndarray, gt: np.ndarray):
    """
    pred, gt: HxW bool
    confmat: dict with keys TP, FP, FN, TN
    """
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, np.logical_not(gt)).sum()
    fn = np.logical_and(np.logical_not(pred), gt).sum()
    tn = np.logical_and(np.logical_not(pred), np.logical_not(gt)).sum()

    confmat["TP"] += int(tp)
    confmat["FP"] += int(fp)
    confmat["FN"] += int(fn)
    confmat["TN"] += int(tn)


def compute_iou_from_confmat(cm):
    denom = cm["TP"] + cm["FP"] + cm["FN"]
    if denom == 0:
        return np.nan
    return cm["TP"] / denom


def compute_pixel_acc_from_confmat(cm):
    denom = cm["TP"] + cm["FP"] + cm["FN"] + cm["TN"]
    if denom == 0:
        return np.nan
    return (cm["TP"] + cm["TN"]) / denom


# ============================================================
# Validation loop
# ============================================================

def validate_lanes(
    model,
    dataloader,
    loss_fn,
    device,
    *,
    logger=None,
    step=None,
    dataset_name=None,
    vis_count: int = 25,
    alpha: float = 0.5,
    pred_threshold: float = 0.0,
    pred_use_sigmoid: bool = False,
):
    """
    Lane segmentation validation with:
      - loss
      - IoU per class (egoleft / egoright / other)
      - mean IoU
      - pixel accuracy
      - visualizations (raw + overlay)

    Returns:
        val_loss (float)
        mean_iou (float)
        pixel_acc (float)
        class_iou_dict (dict)
        vis_images (list[np.ndarray])
    """
    # TODO: Deepak

    model.eval()
    total_loss = 0.0
    num_batches = 0

    confmats = None  # initialized after first forward (depends on C)
    class_names = None

    vis_images = []
    vis_done = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="[Validation Lanes]", leave=False):
            images = batch["image"].to(device, non_blocking=True)  # [B,3,H,W]
            gt_full = batch["gt"].to(device, non_blocking=True)   # [B,C,H',W']

            logits = model(images)  # [B,C,H',W']
            loss = loss_fn(logits, gt_full)

            # Channel count
            C = int(logits.shape[1])
            if confmats is None:
                confmats = [{"TP": 0, "FP": 0, "FN": 0, "TN": 0} for _ in range(C)]
                if C == 3:
                    class_names = ["egoleft", "egoright", "other"]
                elif C == 8:
                    class_names = LANE_CLASS_NAMES_8
                else:
                    class_names = [f"class_{i}" for i in range(C)]

            # -------------------------------------------------
            # Downsample GT for metrics + visualization
            # -------------------------------------------------
            gt = gt_full.float()
            for _ in range(loss_fn.downsample_factor // 2):
                gt = F.max_pool2d(gt, kernel_size=2, stride=2)

            # Binarize GT per channel
            gt_bin = gt > 0.5  # [B,C,H',W']

            total_loss += float(loss.item())
            num_batches += 1

            # -------------------------------------------------
            # Metrics
            # -------------------------------------------------
            if pred_use_sigmoid:
                pred_scores = torch.sigmoid(logits)
            else:
                pred_scores = logits

            preds_bin = pred_scores > pred_threshold  # [B,C,H',W']

            preds_np = preds_bin.cpu().numpy()
            gt_np = gt_bin.cpu().numpy()

            B = preds_np.shape[0]
            for b in range(B):
                for c in range(C):
                    update_binary_confmat(
                        confmats[c],
                        pred=preds_np[b, c],
                        gt=gt_np[b, c],
                    )

            # -------------------------------------------------
            # Visuals
            # -------------------------------------------------
            if vis_done < vis_count and C in (3, 8):
                for b in range(images.shape[0]):
                    if vis_done >= vis_count:
                        break
                    tile = make_lane_vis_pair_C_class(
                        image_chw=images[b],
                        pred_logits_chw=logits[b],
                        gt_maskC_chw=gt_bin[b],
                        alpha=alpha,
                        pred_threshold=pred_threshold,
                        pred_use_sigmoid=pred_use_sigmoid,
                        pred_use_argmax=True if C == 8 else False,
                    )
                    vis_images.append(tile)
                    vis_done += 1

    # -------------------------------------------------
    # Final metrics
    # -------------------------------------------------
    val_loss = total_loss / max(1, num_batches)

    class_iou = {}
    class_acc = {}
    for name, cm in zip(class_names, confmats):
        class_iou[name] = float(compute_iou_from_confmat(cm))
        class_acc[name] = float(compute_pixel_acc_from_confmat(cm))

    mean_iou = float(np.nanmean(list(class_iou.values())))
    pixel_acc = float(np.nanmean(list(class_acc.values())))

    results = {
        "loss": val_loss,
        "mean_iou": mean_iou,
        "pixel_acc": pixel_acc,
        "class_iou": class_iou,
        "class_acc": class_acc,
    }

    if logger is not None and hasattr(logger, "log_validation_lanes"):
        logger.log_validation_lanes(
            step=step,
            dataset=dataset_name,
            val_loss=val_loss,
            mean_iou=mean_iou,
            pixel_acc=pixel_acc,
            class_iou=class_iou,
            class_acc=class_acc,
            vis_images=vis_images,
        )
        return results

    return results, vis_images
    
