"""
Standalone ONNX inference for EgoLanesLite.

No PyTorch, no segmentation_models_pytorch, no timm, no project imports.
Only needs: onnxruntime (or onnxruntime-gpu), numpy, opencv-python, pillow.

Supports both image directory and video input.

Prerequisite: export your PyTorch checkpoint to ONNX first:

    python -m Models.exports.convert_pytorch_to_onnx \\
        -n EgoLanesLite \\
        -c Models/config/EgoLanesLite.yaml \\
        -p runs/training/.../checkpoints/best.pth \\
        -o ego_lanes_lite.onnx

Then run standalone inference:

    python ego_lanes_lite_infer_standalone.py \\
        --model ego_lanes_lite.onnx \\
        --input_dir /path/to/images \\
        --output_dir results/

    python ego_lanes_lite_infer_standalone.py \\
        --model ego_lanes_lite.onnx \\
        --input_video driving.mp4 \\
        --output_dir results/

To use with TensorRT execution provider (if onnxruntime-gpu is installed with
TensorRT support), it is selected automatically when available.
"""

import argparse
import glob
import json
import os
import time
from typing import Any, List, Tuple

import cv2
import numpy as np
from PIL import Image

import onnxruntime as ort
import torch

# ============================================================
# Constants (matches training pipeline)
# ============================================================

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

LANE_COLORS_RGB_3CLASS = {
    "egoleft": (0, 255, 255),
    "egoright": (255, 0, 200),
    "other": (0, 255, 145),
}

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
    # Brighter, high-contrast palette for inference overlays.
    (0, 128, 255),  # bright blue
    (0, 255, 255),    # cyan
    (255, 255, 0),    # yellow
    (255, 0, 255),    # magenta
    (0, 255, 0),      # green
    (255, 165, 0),    # orange
    (255, 0, 0),      # red
    (255, 255, 255),    # white
]

# ============================================================
# Preprocessing
# ============================================================


def preprocess(
    img_rgb: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    """
    Resize, normalize (ImageNet), HWC->CHW, add batch dim.
    Returns float32 array of shape [1, 3, H, W].
    """
    img = cv2.resize(img_rgb, (width, height), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    return img[np.newaxis, ...]  # add batch dim


# ============================================================
# Postprocessing / visualization
# ============================================================


def denorm_image_chw_to_uint8(image_chw: np.ndarray) -> np.ndarray:
    """Reverse ImageNet normalization on a CHW float32 array -> HWC uint8."""
    img = image_chw.transpose(1, 2, 0)  # HWC
    img = (img * IMAGENET_STD + IMAGENET_MEAN) * 255.0
    return np.clip(img, 0, 255).astype(np.uint8)


def logits_to_mask(logits_chw: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """
    logits_chw: [C, H, W] raw logits.
    Returns HxWxC bool mask (independent per channel).
    """
    return (logits_chw > threshold).transpose(1, 2, 0)


def clean_and_fit_lanes(
    binary_mask: np.ndarray,
    *,
    min_area: int = 100,
    poly_degree: int = 2,
) -> List[np.ndarray]:
    """
    Takes a boolean/0-1 mask (HxW), filters small blobs, fits x=f(y) polylines.
    Returns a list of polylines, each shaped [N, 1, 2] int32 for cv2.polylines.
    """
    if binary_mask.dtype != np.bool_:
        binary_mask = binary_mask.astype(bool)

    mask_uint8 = binary_mask.astype(np.uint8) * 255
    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        mask_uint8, connectivity=8
    )

    polylines: List[np.ndarray] = []
    h, w = binary_mask.shape[:2]

    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < int(min_area):
            continue

        y_coords, x_coords = np.where(labels == i)
        if y_coords.size < max(poly_degree + 1, 3):
            continue

        # Fit polynomial: x = f(y)
        poly_coeffs = np.polyfit(y_coords, x_coords, poly_degree)
        poly_func = np.poly1d(poly_coeffs)

        y_min, y_max = int(y_coords.min()), int(y_coords.max())
        y_fit = np.arange(y_min, y_max + 1, dtype=np.int32)
        x_fit = poly_func(y_fit).astype(np.int32)

        valid = (x_fit >= 0) & (x_fit < w) & (y_fit >= 0) & (y_fit < h)
        x_fit = x_fit[valid]
        y_fit = y_fit[valid]
        if x_fit.size < 2:
            continue

        pts = np.stack([x_fit, y_fit], axis=1).astype(np.int32).reshape(-1, 1, 2)
        polylines.append(pts)

    return polylines


def fit_lanes_per_class(
    mask_hwc: np.ndarray,
    min_area: int,
    poly_degree: int,
) -> dict:
    """
    mask_hwc: HxWxC bool mask.
    Returns dict[class_name] = list[polyline], where polyline is list[[x,y],...]
    """
    c = int(mask_hwc.shape[2])
    names = get_class_names(c)
    out: dict = {}
    for idx, name in enumerate(names):
        polys = clean_and_fit_lanes(mask_hwc[..., idx], min_area=min_area, poly_degree=poly_degree)
        if not polys:
            continue
        # Convert to JSON-serializable lists of [x,y]
        out[name] = [p.reshape(-1, 2).tolist() for p in polys]
    return out


def draw_lane_polylines(
    canvas_rgb: np.ndarray,
    mask_hwc: np.ndarray,
    *,
    min_area: int = 100,
    poly_degree: int = 2,
    thickness: int = 3,
) -> np.ndarray:
    """
    Draw fitted lane polylines on an RGB image.
    """
    out = canvas_rgb.copy()
    c = int(mask_hwc.shape[2])

    if c == 8:
        colors = LANE_CLASS_COLORS_RGB_8
        names = LANE_CLASS_NAMES_8
    elif c == 3:
        colors = list(LANE_COLORS_RGB_3CLASS.values())
        names = ["egoleft", "egoright", "other"]
    else:
        colors = [(255, 255, 255)] * c
        names = [f"class_{i}" for i in range(c)]

    for idx in range(c):
        polys = clean_and_fit_lanes(mask_hwc[..., idx], min_area=min_area, poly_degree=poly_degree)
        if not polys:
            continue
        color = colors[idx % len(colors)]
        cv2.polylines(out, polys, isClosed=False, color=color, thickness=int(thickness))

    return out


def apply_lane_colors_3class(canvas: np.ndarray, mask_hwc: np.ndarray) -> np.ndarray:
    out = canvas.copy()
    colors = list(LANE_COLORS_RGB_3CLASS.values())
    for c in range(min(3, mask_hwc.shape[2])):
        ys, xs = np.where(mask_hwc[..., c])
        out[ys, xs, :] = colors[c]
    return out


def apply_lane_colors_8class(canvas: np.ndarray, mask_hwc: np.ndarray) -> np.ndarray:
    out = canvas.copy()
    for c in range(min(8, mask_hwc.shape[2])):
        ys, xs = np.where(mask_hwc[..., c])
        out[ys, xs, :] = LANE_CLASS_COLORS_RGB_8[c]
    return out


def get_class_names(num_classes: int) -> List[str]:
    if num_classes == 3:
        return ["egoleft", "egoright", "other"]
    if num_classes == 8:
        return LANE_CLASS_NAMES_8
    return [f"class_{i}" for i in range(num_classes)]


def detect_classes_in_mask(
    mask_hwc: np.ndarray,
    min_class_pixels: int,
) -> Tuple[List[str], dict]:
    """
    Return detected class names and per-class pixel counts, filtered by min pixels.
    """
    num_classes = int(mask_hwc.shape[2])
    class_names = get_class_names(num_classes)
    pixel_counts = mask_hwc.reshape(-1, num_classes).sum(axis=0).astype(int)

    detected = []
    counts = {}
    for idx, name in enumerate(class_names):
        count = int(pixel_counts[idx])
        if count >= min_class_pixels:
            detected.append(name)
            counts[name] = count
    return detected, counts


def visualize_prediction(
    input_chw: np.ndarray,
    logits_chw: np.ndarray,
    threshold: float = 0.0,
    *,
    render: str = "both",
    lane_min_area: int = 100,
    lane_poly_degree: int = 2,
) -> np.ndarray:
    """
    Build an RGB overlay image (input blended with colored lane predictions).
    Returns HxWx3 uint8 RGB.
    """
    base_img = denorm_image_chw_to_uint8(input_chw)
    mask = logits_to_mask(logits_chw, threshold)
    C = logits_chw.shape[0]

    if C in (3, 8):
        if render == "mask":
            colored = apply_lane_colors_3class(base_img, mask) if C == 3 else apply_lane_colors_8class(base_img, mask)
        elif render == "lines":
            colored = draw_lane_polylines(
                base_img,
                mask,
                min_area=lane_min_area,
                poly_degree=lane_poly_degree,
            )
        elif render == "both":
            mask_colored = apply_lane_colors_3class(base_img, mask) if C == 3 else apply_lane_colors_8class(base_img, mask)
            colored = draw_lane_polylines(
                mask_colored,
                mask,
                min_area=lane_min_area,
                poly_degree=lane_poly_degree,
            )
        else:
            raise RuntimeError("render must be one of: mask, lines, both")
    else:
        seg = logits_chw[0].astype(np.float32)
        if np.isfinite(seg).any():
            vmin, vmax = np.percentile(seg[np.isfinite(seg)], [1, 99])
            vis = np.clip((seg - vmin) / max(vmax - vmin, 1e-8), 0, 1) * 255
        else:
            vis = np.zeros_like(seg)
        return cv2.cvtColor(vis.astype(np.uint8), cv2.COLOR_GRAY2RGB)

    return cv2.addWeighted(colored, 0.5, base_img, 0.5, 0)


# ============================================================
# ONNX session setup
# ============================================================


def create_onnx_session(model_path: str) -> ort.InferenceSession:
    available = ort.get_available_providers()
    print(f"[ONNX Runtime] Available providers: {available}")
    providers = []
    for p in ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]:
        if p in available:
            providers.append(p)
    if not providers:
        providers = ["CPUExecutionProvider"]

    sess = ort.InferenceSession(model_path, providers=providers)
    active = sess.get_providers()
    print(f"[ONNX Runtime] Active providers: {active}")
    return sess


def create_torchscript_session(model_path: str, device: str) -> tuple[torch.jit.ScriptModule, torch.device]:
    torch_device = torch.device(device)
    print(f"[TorchScript] Loading model on device: {torch_device}")
    model = torch.jit.load(model_path, map_location=torch_device)
    model = model.eval().to(torch_device)
    return model, torch_device


def create_backend_session(
    model_path: str,
    backend: str,
    torch_device: str,
) -> tuple[str, Any]:
    if backend == "onnx":
        return "onnx", create_onnx_session(model_path)
    if backend == "torchscript":
        return "torchscript", create_torchscript_session(model_path, torch_device)
    raise RuntimeError(f"Unsupported backend: {backend}")


def infer_logits_chw(session_backend: str, session_obj: Any, input_array: np.ndarray) -> np.ndarray:
    if session_backend == "onnx":
        sess: ort.InferenceSession = session_obj
        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name
        result = sess.run([output_name], {input_name: input_array})
        return result[0][0]  # [C, H, W]

    model, torch_device = session_obj
    with torch.no_grad():
        inp = torch.from_numpy(input_array).to(torch_device)
        out = model(inp)
        if not isinstance(out, torch.Tensor):
            raise RuntimeError(f"TorchScript model output must be a Tensor, got: {type(out)}")
        return out.detach().cpu().numpy()[0]  # [C, H, W]


def synchronize_backend(session_backend: str, session_obj: Any) -> None:
    """
    Synchronize backend for accurate latency timing when needed.
    """
    if session_backend == "torchscript":
        _model, torch_device = session_obj
        if torch_device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()


def benchmark_raw_inference(
    session_backend: str,
    session_obj: Any,
    input_array: np.ndarray,
    *,
    warmup_runs: int,
    benchmark_runs: int,
) -> None:
    """
    Benchmark model-only inference on one preprocessed [1,3,H,W] input.
    Excludes disk I/O, video decode, visualization, and file writing.
    """
    if benchmark_runs <= 0:
        raise RuntimeError("--benchmark_runs must be > 0")
    if warmup_runs < 0:
        raise RuntimeError("--benchmark_warmup must be >= 0")

    print(
        f"[Benchmark] backend={session_backend}, "
        f"warmup_runs={warmup_runs}, benchmark_runs={benchmark_runs}"
    )

    for _ in range(warmup_runs):
        _ = infer_logits_chw(session_backend, session_obj, input_array)
    synchronize_backend(session_backend, session_obj)

    t0 = time.perf_counter()
    for _ in range(benchmark_runs):
        _ = infer_logits_chw(session_backend, session_obj, input_array)
    synchronize_backend(session_backend, session_obj)
    t1 = time.perf_counter()

    total_s = max(t1 - t0, 1e-12)
    avg_ms = (total_s / benchmark_runs) * 1000.0
    fps = benchmark_runs / total_s

    print(f"[Benchmark] Total time: {total_s:.4f} s")
    print(f"[Benchmark] Avg inference latency: {avg_ms:.3f} ms")
    print(f"[Benchmark] Throughput: {fps:.2f} FPS")


def run_session(sess: ort.InferenceSession, input_array: np.ndarray) -> np.ndarray:
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    result = sess.run([output_name], {input_name: input_array})
    return result[0]  # [B, C, H, W]


# ============================================================
# Inference: images
# ============================================================


def list_images(input_dir: str) -> List[str]:
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    files: List[str] = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(input_dir, ext)))
    files.sort()
    return files


def run_inference_images(
    session_backend: str,
    session_obj: Any,
    image_paths: List[str],
    output_dir: str,
    height: int,
    width: int,
    threshold: float,
    min_class_pixels: int,
    lane_min_area: int,
    lane_poly_degree: int,
    render: str,
    save_lane_lines_json: bool,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    print(f"Running inference on {len(image_paths)} images...")
    jsonl_path = os.path.join(output_dir, "frame_lane_classes.jsonl")

    with open(jsonl_path, "w", encoding="utf-8") as jsonl_file:
        for idx, img_path in enumerate(image_paths):
            img_rgb = np.array(Image.open(img_path).convert("RGB"))
            input_arr = preprocess(img_rgb, height, width)

            logits_chw = infer_logits_chw(session_backend, session_obj, input_arr)
            input_chw = input_arr[0]  # [3, H, W]

            overlay = visualize_prediction(
                input_chw,
                logits_chw,
                threshold,
                render=render,
                lane_min_area=lane_min_area,
                lane_poly_degree=lane_poly_degree,
            )
            mask_hwc = logits_to_mask(logits_chw, threshold)
            detected, pixel_counts = detect_classes_in_mask(mask_hwc, min_class_pixels)
            lane_lines = (
                fit_lanes_per_class(mask_hwc, lane_min_area, lane_poly_degree) if save_lane_lines_json else {}
            )

            base_img = denorm_image_chw_to_uint8(input_chw)
            tile = np.concatenate([base_img, overlay], axis=1)

            base = os.path.splitext(os.path.basename(img_path))[0]
            vis_path = os.path.join(output_dir, f"{base}_vis.png")
            cv2.imwrite(vis_path, cv2.cvtColor(tile, cv2.COLOR_RGB2BGR))
            print(f"  {img_path} -> {vis_path}")

            record = {
                "frame_index": idx,
                "source": img_path,
                "detected_lane_classes": detected,
                "pixel_counts": pixel_counts,
                "lane_lines": lane_lines,
            }
            jsonl_file.write(json.dumps(record) + "\n")

    print(f"Saved per-frame lane classes JSONL: {jsonl_path}")


# ============================================================
# Inference: video
# ============================================================


def run_inference_video(
    session_backend: str,
    session_obj: Any,
    video_path: str,
    output_dir: str,
    height: int,
    width: int,
    threshold: float,
    min_class_pixels: int,
    lane_min_area: int,
    lane_poly_degree: int,
    render: str,
    save_lane_lines_json: bool,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(video_path))[0]
    out_video_path = os.path.join(output_dir, f"{base}_vis.mp4")
    jsonl_path = os.path.join(output_dir, f"{base}_lane_classes.jsonl")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps_in = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    fps_out = fps_in if np.isfinite(fps_in) and fps_in > 1e-3 else 30.0

    print(f"Running video inference: {video_path} -> {out_video_path}")

    writer = None
    frame_count = 0
    t_start = time.time()
    t_last_log = t_start
    frames_since_log = 0

    with open(jsonl_path, "w", encoding="utf-8") as jsonl_file:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            input_arr = preprocess(frame_rgb, height, width)

            logits_chw = infer_logits_chw(session_backend, session_obj, input_arr)
            overlay = visualize_prediction(
                input_arr[0],
                logits_chw,
                threshold,
                render=render,
                lane_min_area=lane_min_area,
                lane_poly_degree=lane_poly_degree,
            )
            mask_hwc = logits_to_mask(logits_chw, threshold)
            detected, pixel_counts = detect_classes_in_mask(mask_hwc, min_class_pixels)
            lane_lines = (
                fit_lanes_per_class(mask_hwc, lane_min_area, lane_poly_degree) if save_lane_lines_json else {}
            )

            record = {
                "frame_index": frame_count,
                "detected_lane_classes": detected,
                "pixel_counts": pixel_counts,
                "lane_lines": lane_lines,
            }
            jsonl_file.write(json.dumps(record) + "\n")

            frame_out = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

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

            if frames_since_log >= 30:
                now = time.time()
                dt = max(now - t_last_log, 1e-6)
                live_fps = frames_since_log / dt
                print(f"  Processed {frame_count} frames ({live_fps:.2f} FPS)")
                t_last_log = now
                frames_since_log = 0

    cap.release()
    if writer is not None:
        writer.release()

    elapsed = max(time.time() - t_start, 1e-6)
    print(
        f"Done: {frame_count} frames in {elapsed:.2f}s "
        f"({frame_count / elapsed:.2f} FPS avg, output FPS={fps_out:.2f})"
    )
    print(f"Saved per-frame lane classes JSONL: {jsonl_path}")


# ============================================================
# CLI
# ============================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Standalone ONNX inference for EgoLanesLite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "-m", "--model", required=True,
        help="Path to model file (.onnx for ONNX Runtime, .pt for TorchScript)",
    )
    parser.add_argument(
        "--backend", choices=["auto", "onnx", "torchscript"], default="auto",
        help="Inference backend selection (default: auto by model extension)",
    )
    parser.add_argument(
        "--torch_device", default="cuda",
        help="Torch device for TorchScript backend (default: cuda)",
    )
    parser.add_argument(
        "-i", "--input_dir", default=None,
        help="Directory with input images (png/jpg). Use this OR --input_video.",
    )
    parser.add_argument(
        "-v", "--input_video", default=None,
        help="Path to an input video file. Use this OR --input_dir.",
    )
    parser.add_argument(
        "-o", "--output_dir", default="runs/inference/EgoLanesLite_onnx",
        help="Directory where predictions will be saved",
    )
    parser.add_argument(
        "--height", type=int, default=416,
        help="Input height (must match ONNX export resolution)",
    )
    parser.add_argument(
        "--width", type=int, default=800,
        help="Input width (must match ONNX export resolution)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.0,
        help="Logit threshold for lane detection (default: 0.0)",
    )
    parser.add_argument(
        "--min_class_pixels", type=int, default=250,
        help="Ignore classes with fewer than this many pixels per frame",
    )
    parser.add_argument(
        "--lane_min_area", type=int, default=250,
        help="Min connected-component pixel area for lane line fitting (default: 100)",
    )
    parser.add_argument(
        "--lane_poly_degree", type=int, default=2,
        help="Polynomial degree for lane fitting x=f(y) (default: 2)",
    )
    parser.add_argument(
        "--render", choices=["mask", "lines", "both"], default="lines",
        help="Visualization mode (default: lines)",
    )
    parser.add_argument(
        "--save_lane_lines_json", default=True, type=bool,
        help="Include fitted lane polylines in the JSONL output (can be large)",
    )
    parser.add_argument(
        "--benchmark_only", action="store_true",
        help="Run raw model benchmark on one image and exit (no output video/images)",
    )
    parser.add_argument(
        "--benchmark_image", default=None,
        help="Image path for benchmark input. If omitted, first image from --input_dir is used.",
    )
    parser.add_argument(
        "--benchmark_warmup", type=int, default=5,
        help="Warmup iterations before benchmark timing (default: 5)",
    )
    parser.add_argument(
        "--benchmark_runs", type=int, default=200,
        help="Timed inference iterations for benchmark (default: 200)",
    )

    args = parser.parse_args()

    if not args.benchmark_only:
        if args.input_dir is not None and args.input_video is not None:
            raise RuntimeError("Specify only one of --input_dir or --input_video.")
        if args.input_dir is None and args.input_video is None:
            raise RuntimeError("You must specify either --input_dir or --input_video.")

    backend = args.backend
    if backend == "auto":
        if args.model.lower().endswith(".onnx"):
            backend = "onnx"
        elif args.model.lower().endswith(".pt"):
            backend = "torchscript"
        else:
            raise RuntimeError("Could not infer backend. Use --backend onnx|torchscript.")

    if backend == "torchscript" and args.torch_device == "cuda" and not torch.cuda.is_available():
        print("[TorchScript] CUDA not available; falling back to CPU.")
        args.torch_device = "cpu"

    session_backend, session_obj = create_backend_session(args.model, backend, args.torch_device)

    if args.benchmark_only:
        bench_image = args.benchmark_image
        if bench_image is None:
            if args.input_dir is None:
                raise RuntimeError(
                    "For --benchmark_only, pass --benchmark_image or provide --input_dir."
                )
            image_paths = list_images(args.input_dir)
            if not image_paths:
                raise RuntimeError(f"No images found in: {args.input_dir}")
            bench_image = image_paths[0]

        print(f"[Benchmark] Using input image: {bench_image}")
        img_rgb = np.array(Image.open(bench_image).convert("RGB"))
        input_arr = preprocess(img_rgb, args.height, args.width)

        benchmark_raw_inference(
            session_backend,
            session_obj,
            input_arr,
            warmup_runs=args.benchmark_warmup,
            benchmark_runs=args.benchmark_runs,
        )
        return

    if args.input_video is not None:
        run_inference_video(
            session_backend, session_obj, args.input_video, args.output_dir,
            args.height, args.width, args.threshold, args.min_class_pixels,
            args.lane_min_area, args.lane_poly_degree, args.render, args.save_lane_lines_json,
        )
    else:
        image_paths = list_images(args.input_dir)
        if not image_paths:
            raise RuntimeError(f"No images found in: {args.input_dir}")
        run_inference_images(
            session_backend, session_obj, image_paths, args.output_dir,
            args.height, args.width, args.threshold, args.min_class_pixels,
            args.lane_min_area, args.lane_poly_degree, args.render, args.save_lane_lines_json,
        )


if __name__ == "__main__":
    main()
