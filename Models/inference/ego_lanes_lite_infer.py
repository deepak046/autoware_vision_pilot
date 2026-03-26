"""
Lite ego-lanes inference script.

This mirrors the lite training stack (`LiteTrainerBase` + `EgoLanesLiteTrainer`)
but only builds the model, loads a checkpoint, and runs inference on images.

Head output shapes:
  - 3 channels: legacy egoleft / egoright / other visualization.
  - 8 channels: lane classes (order matches `helpers.lanes.LANE_CLASS_NAMES_8`);
    default is argmax per pixel + sigmoid; use ``--lane8-no-argmax`` for
    independent per-channel thresholds.

Usage (from repository root):

    python -m Models.inference.ego_lanes_infer \\
        --config Models/config/EgoLanesLite.yaml \\
        --checkpoint /path/to/checkpoints/best.pth \\
        --input_dir /path/to/images \\
        --output_dir runs/inference/EgoLanesLite
"""

import argparse
import os
import glob
import time
from typing import List

import cv2
import numpy as np
from PIL import Image

import torch
from torchvision import transforms

from Models.data_utils.lite_models.helpers.lanes import (
    _apply_lane_colors_rgb,
    apply_lane_colors_rgb_8class,
    denorm_image_chw_to_uint8,
    logits_to_lane_class_id_hw,
    logits_to_lane_mask3_argmax,
    logits_to_lane_mask3,
    logits_to_lane_mask8,
    logits_to_lane_mask8_argmax,
)
from Models.data_utils.lite_models.helpers.training import load_yaml, set_global_seed
from Models.training.lite_trainer_base import LiteTrainerBase


class EgoLanesLiteInferModel(LiteTrainerBase):
    """
    Thin wrapper around `LiteTrainerBase` that:
      - builds the encoder+decoder stack
      - loads a checkpoint (model_state only)
      - exposes `self.model` in eval mode on the proper device
    """

    def __init__(self, cfg: dict, checkpoint_path: str, device: str | None = None):
        super().__init__(cfg)

        # Optional device override
        if device is not None:
            self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Needed by _build_model_stack / _maybe_resume
        self.network_cfg = cfg["network"]
        self.backbone_cfg = self.network_cfg.get("backbone", {})
        self.decoder_cfg = self.network_cfg.get("decoder", {})
        self.head_cfg = self.network_cfg.get("head", {})

        # Match training behavior for backbone name normalization
        if "timm" not in self.backbone_cfg["type"]:
            self.backbone_cfg["type"] = "timm-" + self.backbone_cfg["type"].replace("_", "-")

        # Build model
        self._build_model_stack()

        # Prepare a minimal checkpoint config and force "val" mode so that
        # _maybe_resume loads only model weights and skips optimizer/scheduler.
        self.exp_name = "val"
        self.ckpt_cfg = {
            "load_from": checkpoint_path,
            "strict_load": True,
            "fine_tune": False,
        }

        self._maybe_resume()

        self.model.eval()
        print(f"[EgoLanesLiteInferModel] Loaded checkpoint from: {checkpoint_path}")


def build_transform() -> transforms.Compose:
    """
    Standard ImageNet-style normalization, matching typical encoder expectations.
    """
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            transforms.Resize((416, 800)),

        ]
    )


def list_images(input_dir: str) -> List[str]:
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    files: List[str] = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(input_dir, ext)))
    files.sort()
    return files


def run_inference(
    model_wrapper: EgoLanesLiteInferModel,
    image_paths: List[str],
    output_dir: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    device = model_wrapper.device
    model = model_wrapper.model
    transform = build_transform()

    print(f"Running inference on {len(image_paths)} images...")

    with torch.no_grad():
        for img_path in image_paths:
            img = Image.open(img_path).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(device)

            logits = model(tensor)
            # logits: [1, C, H, W]

            # Convert to numpy for saving
            logits_np = logits.squeeze(0).cpu().numpy()  # [C, H, W] or [H, W] if C==1

            base = os.path.splitext(os.path.basename(img_path))[0]

            # Save raw logits as .npy for downstream processing
            # npy_path = os.path.join(output_dir, f"{base}_logits.npy")
            # np.save(npy_path, logits_np)

            # Visualization: input image + pred logits (no GT)
            image_chw = tensor.squeeze(0)
            pred_logits_chw = logits.squeeze(0)

            if pred_logits_chw.shape[0] == 3:
                # 3-channel lane model (egoleft, egoright, other). Use argmax so
                # each pixel gets one class; avoids "all green" when the model
                # predicts "other" with high prob everywhere (e.g. dice+focal+edge).
                base_img = denorm_image_chw_to_uint8(image_chw)
                pred_mask3 = logits_to_lane_mask3(
                    pred_logits_chw, threshold=0.0, use_sigmoid=False
                )
                pred_colored = _apply_lane_colors_rgb(base_img, pred_mask3)
                pred_overlay = cv2.addWeighted(pred_colored, 0.5, base_img, 0.5, 0)
                tile = np.concatenate([base_img, pred_overlay], axis=1)
                vis_path = os.path.join(output_dir, f"{base}_vis.png")
                cv2.imwrite(vis_path, cv2.cvtColor(tile, cv2.COLOR_RGB2BGR))
            elif pred_logits_chw.shape[0] == 8:
                # 8-class lane taxonomy (same order as LANE_CLASS_NAMES_8)
                base_img = denorm_image_chw_to_uint8(image_chw)
                pred_mask8 = logits_to_lane_mask8(
                    pred_logits_chw,
                    threshold=0.0,
                    use_sigmoid=False,
                )
                pred_colored = apply_lane_colors_rgb_8class(base_img, pred_mask8)
                pred_overlay = cv2.addWeighted(pred_colored, 0.5, base_img, 0.5, 0)
                tile = np.concatenate([base_img, pred_overlay], axis=1)
                vis_path = os.path.join(output_dir, f"{base}_vis.png")
                cv2.imwrite(vis_path, cv2.cvtColor(tile, cv2.COLOR_RGB2BGR))
            else:
                # Fallback for non-3-channel output
                seg = logits_np.squeeze()
                if seg.ndim == 3:
                    seg = seg[0]
                seg = seg.astype(np.float32)
                if np.isfinite(seg).any():
                    vmin, vmax = np.percentile(seg[np.isfinite(seg)], [1, 99])
                    vis = (
                        np.clip((seg - vmin) / max(vmax - vmin, 1e-8), 0, 1) * 255
                    ).astype(np.uint8)
                else:
                    vis = np.zeros_like(seg, dtype=np.uint8)
                vis_path = os.path.join(output_dir, f"{base}_vis.png")
                Image.fromarray(vis).save(vis_path)

            print(f"Saved prediction for {img_path} -> {vis_path}")


def run_inference_video(
    model_wrapper: EgoLanesLiteInferModel,
    video_path: str,
    output_dir: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    device = model_wrapper.device
    model = model_wrapper.model
    transform = build_transform()

    base = os.path.splitext(os.path.basename(video_path))[0]
    out_video_path = os.path.join(output_dir, f"{base}_vis.mp4")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    print(f"Running video inference on {video_path} -> {out_video_path}")

    frame_count = 0
    t_start = time.time()
    t_last_log = t_start
    frames_since_log = 0

    # IMPORTANT: Write frames as we go (streaming) to avoid RAM blowups on long videos.
    # Output FPS is set to the input video FPS (fallback to 30 if unknown).
    fps_in = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    fps_out = fps_in if np.isfinite(fps_in) and fps_in > 1e-3 else 30.0
    writer = None

    with torch.no_grad():
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(frame_rgb)
            tensor = transform(img_pil).unsqueeze(0).to(device)

            logits = model(tensor)

            image_chw = tensor.squeeze(0)
            pred_logits_chw = logits.squeeze(0)

            if pred_logits_chw.shape[0] == 3:
                base_img = denorm_image_chw_to_uint8(image_chw)
                pred_mask3 = logits_to_lane_mask3(
                    pred_logits_chw, threshold=0.0, use_sigmoid=False
                )
                pred_colored = _apply_lane_colors_rgb(base_img, pred_mask3)
                # Only the processed (overlay) image in the video
                tile_rgb = cv2.addWeighted(pred_colored, 0.5, base_img, 0.5, 0)
            elif pred_logits_chw.shape[0] == 8:
                base_img = denorm_image_chw_to_uint8(image_chw)
                pred_mask8 = logits_to_lane_mask8(
                    pred_logits_chw, threshold=0.0, use_sigmoid=False
                )
                pred_colored = apply_lane_colors_rgb_8class(base_img, pred_mask8)
                # Only the processed (overlay) image in the video
                tile_rgb = cv2.addWeighted(pred_colored, 0.5, base_img, 0.5, 0)         
            else:
                seg = logits.squeeze(0).detach().cpu().numpy()
                if seg.ndim == 3:
                    seg = seg[0]
                seg = seg.astype(np.float32)
                if np.isfinite(seg).any():
                    vmin, vmax = np.percentile(seg[np.isfinite(seg)], [1, 99])
                    vis = (
                        np.clip((seg - vmin) / max(vmax - vmin, 1e-8), 0, 1) * 255
                    ).astype(np.uint8)
                else:
                    vis = np.zeros_like(seg, dtype=np.uint8)
                tile_rgb = cv2.cvtColor(vis, cv2.COLOR_GRAY2RGB)

            frame_out = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2BGR)

            if writer is None:
                h, w = frame_out.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(out_video_path, fourcc, fps_out, (w, h))
                if not writer.isOpened():
                    cap.release()
                    raise RuntimeError(f"Could not open VideoWriter: {out_video_path}")

            writer.write(frame_out)
            frame_count += 1
            frames_since_log += 1

            # Live processing rate (running FPS, printed every 30 frames)
            if frames_since_log >= 30:
                now = time.time()
                dt = max(now - t_last_log, 1e-6)
                live_fps = frames_since_log / dt
                print(f"[Video inference] Processed {frame_count} frames "
                      f"({live_fps:.2f} FPS over last {frames_since_log} frames)")
                t_last_log = now
                frames_since_log = 0

    cap.release()
    if writer is not None:
        writer.release()

    t_end = time.time()
    elapsed = max(t_end - t_start, 1e-6)
    proc_fps = frame_count / elapsed

    print(f"[Video inference] Final: processed {frame_count} frames in {elapsed:.2f}s "
          f"-> {proc_fps:.2f} FPS average (output video FPS = {fps_out:.2f})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Lite Ego Lanes inference")
    parser.add_argument(
        "-c",
        "--config",
        default="Models/config/EgoLanesLite.yaml",
        help="Path to lane detection lite YAML config used for training",
    )
    parser.add_argument(
        "-k",
        "--checkpoint",
        required=True,
        help="Path to trained lite checkpoint (e.g., best.pth or last.pth)",
    )
    parser.add_argument(
        "-i",
        "--input_dir",
        required=False,
        help="Directory with input images (png/jpg). Use this OR --input_video.",
    )
    parser.add_argument(
        "-v",
        "--input_video",
        required=False,
        help="Path to an input video file. Use this OR --input_dir.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        default="runs/inference/EgoLanesLite",
        help="Directory where predictions will be saved",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional device override, e.g. 'cuda:0' or 'cpu' (default: config/auto)",
    )

    args = parser.parse_args()

    cfg = load_yaml(args.config)
    seed = cfg.get("experiment", {}).get("seed", 42)
    set_global_seed(seed)

    infer_model = EgoLanesLiteInferModel(cfg, checkpoint_path=args.checkpoint, device=args.device)

    if args.input_video is not None and args.input_dir is not None:
        raise RuntimeError("Please specify only one of --input_dir or --input_video.")

    if args.input_video is not None:
        run_inference_video(
            infer_model,
            args.input_video,
            args.output_dir)
    elif args.input_dir is not None:
        image_paths = list_images(args.input_dir)
        if not image_paths:
            raise RuntimeError(f"No images found in directory: {args.input_dir}")
        run_inference(infer_model, image_paths, args.output_dir)
    else:
        raise RuntimeError("You must specify either --input_dir or --input_video.")


if __name__ == "__main__":
    main()
