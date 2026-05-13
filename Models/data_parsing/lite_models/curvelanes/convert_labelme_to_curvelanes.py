import os
import re
import cv2
import json
import glob
import warnings
import argparse
import numpy as np
from collections import Counter, defaultdict

try:
    warnings.simplefilter("ignore", np.exceptions.RankWarning)
except AttributeError:
    warnings.simplefilter("ignore", np.RankWarning)

NUM_LANES = 6
HALF_LANES = NUM_LANES // 2
IMG_WIDTH = 1920
IMG_HEIGHT = 1080
LANE_DRAW_WIDTH = 30
MIN_LANE_LENGTH = 80
ROW_ANCHOR_START = 400 # 64 row anchors
ROW_ANCHOR_END = 1080
ROW_ANCHOR_STEP = 10

# Stable 8-class taxonomy kept for compatibility/debugging.
LANE_CLASSES_8 = [
    "continuous_white_line",   # 0
    "continuous_yellow_line",  # 1
    "dashed_white_line",       # 2
    "double_white_lines",      # 3
    "double_yellow_lines",     # 4
    "curb_line",               # 5
    "stop_line",               # 6
    "invisible_line",          # 7
]

# Stable lane taxonomy for class assignment (requested).
LANE_CLASSES_9 = [
    "continuous_white_line",   # 0
    "continuous_yellow_line",  # 1
    "dashed_white_line",       # 2
    "double_white_lines",      # 3
    "double_yellow_lines",     # 4
    "curb_line",               # 5
    "stop_line",               # 6
    "invisible_line",          # 7
    "Occluded_curb_lines",     # 8
]

LANE_CLASSES_3 = [
    "egoleft",
    "egoright",
    "other_lanes",
]

ACTIVE_LANE_CLASSES = list(LANE_CLASSES_9)
ACTIVE_LANE_CLASS_TO_ID = {name: idx for idx, name in enumerate(ACTIVE_LANE_CLASSES)}

# BGR debug colors (keys match exact JSON label strings in LANE_CLASSES).
LANE_8_CLASS_COLOR_BGR = {
    "continuous_white_line": (240, 240, 240),
    "continuous_yellow_line": (0, 255, 255),
    "dashed_white_line": (200, 200, 200),
    "double_white_lines": (220, 220, 220),
    "double_yellow_lines": (0, 220, 255),
    "curb_line": (0, 140, 255),
    "stop_line": (0, 0, 255),
    "invisible_line": (128, 128, 128),
}

LANE_9_CLASS_COLOR_BGR = {
    "continuous_white_line": (240, 240, 240),
    "continuous_yellow_line": (0, 255, 255),
    "dashed_white_line": (200, 200, 200),
    "double_white_lines": (220, 220, 220),
    "double_yellow_lines": (0, 220, 255),
    "curb_line": (0, 140, 255),
    "stop_line": (0, 0, 255),
    "invisible_line": (128, 128, 128),
    "Occluded_curb_lines": (255, 0, 255),
}


def bgr_color_for_lane_label(label):
    key = str(label).strip()
    if key in LANE_9_CLASS_COLOR_BGR:
        return LANE_9_CLASS_COLOR_BGR[key]
    h = abs(hash(key))
    return (h & 255, (h >> 8) & 255, (h >> 16) & 255)


def _debug_overlay_text(slot, class_label):
    short = str(class_label).strip() if class_label else ""
    if len(short) > 22:
        short = short[:21] + "~"
    if slot is not None:
        return f"{slot}:{short}" if short else str(slot)
    return short or "?"


def build_lane_taxonomy(class_mode: int):
    if class_mode == 9:
        classes = list(LANE_CLASSES_9)
    elif class_mode == 8:
        classes = list(LANE_CLASSES_8)
    elif class_mode == 3:
        classes = list(LANE_CLASSES_3)
    else:
        raise ValueError(f"Unsupported class_mode: {class_mode}")
    return classes, {name: idx for idx, name in enumerate(classes)}

# Thickness modulation: boost lanes near image center.
LANE_THICKNESS_MAX_EXTRA = 10  # pixels added at image center (keeps boundary lanes unchanged)
LANE_THICKNESS_CENTER_BAND = 0.45  # fraction of half-width; outside band -> no boost

# Boundary curb filtering: if two boundary lanes are close and one is curb, keep curb only.
BOUNDARY_EDGE_MARGIN_FRAC = 0.18  # left/right edge zone (fraction of image width)
BOUNDARY_CLOSE_X_THRESH_PX = 45   # pixels; "very close" at boundary

# Stop lines are horizontal and are filtered out of the slot assignment process.
STOP_LINE_MASK_VALUE = 255
STOP_LINE_DRAW_THICKNESS = 18


def _canonical_label(shape: dict) -> str:
    """Label strings in JSON are treated as exact class names (see LANE_CLASSES)."""
    return str(shape.get("label", "")).strip()


def is_stop_line_shape(shape: object) -> bool:
    if not isinstance(shape, dict):
        return False
    if _canonical_label(shape) in SKIP_LABELS:
        return False
    return _canonical_label(shape) == "stop_line"


def _lane_mean_x(points) -> float:
    pts = np.array(points, dtype=np.float32)
    if pts.size == 0:
        return float("nan")
    return float(np.mean(pts[:, 0]))


def assign_three_class_names(shapes, img_width: int):
    """
    Assign per-shape 3-class names:
      - egoleft: closest lane to image center on left side
      - egoright: closest lane to image center on right side
      - other_lanes: all remaining lanes
    """
    class_name_by_shape_idx = {}
    if not shapes or img_width <= 0:
        return class_name_by_shape_idx

    center_x = img_width / 2.0
    candidates = []
    for idx, shape in enumerate(shapes):
        if not isinstance(shape, dict):
            continue
        label = str(shape.get("label", "")).strip()
        if label in SKIP_LABELS:
            continue
        pts = shape.get("points", [])
        if not isinstance(pts, list) or len(pts) < 2:
            continue
        if label == "stop_line":
            class_name_by_shape_idx[idx] = "other_lanes"
            continue
        mx = _lane_mean_x(pts)
        if not np.isfinite(mx):
            class_name_by_shape_idx[idx] = "other_lanes"
            continue
        candidates.append((idx, float(mx)))
        class_name_by_shape_idx[idx] = "other_lanes"

    left_candidates = [(i, mx) for i, mx in candidates if mx < center_x]
    right_candidates = [(i, mx) for i, mx in candidates if mx >= center_x]

    if left_candidates:
        left_idx = max(left_candidates, key=lambda t: t[1])[0]
        class_name_by_shape_idx[left_idx] = "egoleft"
    if right_candidates:
        right_idx = min(right_candidates, key=lambda t: t[1])[0]
        class_name_by_shape_idx[right_idx] = "egoright"

    return class_name_by_shape_idx

def filter_boundary_close_lanes_keep_curb(shapes, img_width: int):
    """
    If there are 2 very close lanes on the boundary and one is a curb_line,
    keep the curb_line lane and remove the other.
    """
    if not shapes:
        return shapes

    edge_margin = int(img_width * BOUNDARY_EDGE_MARGIN_FRAC)
    left_edge = edge_margin
    right_edge = img_width - edge_margin

    entries = []
    for idx, shape in enumerate(shapes):
        if not isinstance(shape, dict):
            continue
        label = shape.get("label", "")
        if label in SKIP_LABELS:
            continue
        pts = shape.get("points", [])
        if not isinstance(pts, list) or len(pts) < 2:
            continue
        mx = _lane_mean_x(pts)
        if not np.isfinite(mx):
            continue
        if mx <= left_edge:
            side = "left"
        elif mx >= right_edge:
            side = "right"
        else:
            side = None
        lane_class = str(label).strip()
        entries.append(
            {
                "idx": idx,
                "shape": shape,
                "mean_x": mx,
                "side": side,
                "lane_class": lane_class,
            }
        )

    keep = [True] * len(shapes)

    for side in ("left", "right"):
        side_entries = [e for e in entries if e["side"] == side]
        side_entries.sort(key=lambda e: e["mean_x"])
        for a, b in zip(side_entries, side_entries[1:]):
            if abs(a["mean_x"] - b["mean_x"]) > BOUNDARY_CLOSE_X_THRESH_PX:
                continue
            # Do not drop stop_line when paired with curb (different semantics).
            if a["lane_class"] == "stop_line" or b["lane_class"] == "stop_line":
                continue
            a_is_curb = a["lane_class"] == "curb_line"
            b_is_curb = b["lane_class"] == "curb_line"
            if a_is_curb and not b_is_curb:
                keep[b["idx"]] = False
            elif b_is_curb and not a_is_curb:
                keep[a["idx"]] = False

    return [s for i, s in enumerate(shapes) if keep[i]]


def lane_thickness_for_points(points, img_width: int, base: int) -> int:
    """
    Increase thickness for lanes close to the center of the image,
    keep boundary lanes at the base thickness.
    """
    mx = _lane_mean_x(points)
    if not np.isfinite(mx) or img_width <= 0:
        return int(base)

    edge_margin = img_width * BOUNDARY_EDGE_MARGIN_FRAC
    if mx <= edge_margin or mx >= (img_width - edge_margin):
        return int(base)

    half_w = img_width / 2.0
    if half_w <= 1e-6:
        return int(base)

    d = abs(mx - half_w) / half_w  # 0 at center, 1 at edges
    if d >= LANE_THICKNESS_CENTER_BAND:
        return int(base)

    # Linear boost within the center band.
    boost = (1.0 - (d / LANE_THICKNESS_CENTER_BAND)) * float(LANE_THICKNESS_MAX_EXTRA)
    return int(max(1, round(base + boost)))


def _fit_degree(n_points, max_deg=3):
    """
    Choose polynomial degree ensuring least-squares smoothing.
    Requires at least deg+2 points so polyfit is overdetermined.
    2-3 pts -> deg 1, 4-5 pts -> deg 2, 6+ pts -> deg 3.
    """
    if n_points < 2:
        return 0
    return min(max(n_points - 2, 1), max_deg)


def _lane_fit(lane):
    """Return (y_min, y_max, poly_coeffs) for a lane, or None if too few points."""
    pts = np.array(lane)
    xs, ys = pts[:, 0], pts[:, 1]
    deg = _fit_degree(len(xs))
    if deg < 1 or (ys.max() - ys.min()) < 1:
        return None
    curve = np.polyfit(ys, xs, deg=deg)
    return ys.min(), ys.max(), curve


def _line_y_at_x(x0, y0, x1, y1, x):
    """Evaluate y at a given x on the line through (x0,y0)-(x1,y1)."""
    dx = x1 - x0
    if abs(dx) < 1e-9:
        return float("inf")
    return y0 + (y1 - y0) / dx * (x - x0)


def _line_x_at_y(x0, y0, x1, y1, y):
    """Evaluate x at a given y on the line through (x0,y0)-(x1,y1)."""
    dy = y1 - y0
    if abs(dy) < 1e-9:
        return float("inf")
    return x0 + (x1 - x0) / dy * (y - y0)


def calc_k(points, height, width, angle=False):
    """
    Compute lane direction/position, adapted from convert_curvelanes.py.
    `points` is a list of [x, y] pairs
    """
    xs = np.array([p[0] for p in points])
    ys = np.array([p[1] for p in points])

    # Lane length filter
    length = np.sqrt((xs[0] - xs[-1]) ** 2 + (ys[0] - ys[-1]) ** 2)
    if length < MIN_LANE_LENGTH:
        return -10

    x_range = xs.max() - xs.min()
    if x_range < 1:
        rad = np.pi / 2 if ys[0] > ys[-1] else -np.pi / 2
    else:
        p = np.polyfit(xs, ys, deg=1)
        rad = np.arctan(p[0])

    if angle:
        return rad

    x0, y0 = xs[-2], ys[-2]
    x1, y1 = xs[-1], ys[-1]

    if rad < 0:
        y = _line_y_at_x(x0, y0, x1, y1, 0)
        if y > height:
            result = _line_x_at_y(x0, y0, x1, y1, height)
        else:
            result = -(height - y)
    else:
        y = _line_y_at_x(x0, y0, x1, y1, width)
        if y > height:
            result = _line_x_at_y(x0, y0, x1, y1, height)
        else:
            result = width + (height - y)

    return result


def interpolate_points(points, num_interp=10):
    """
    For 2-point lines, interpolate to create a dense polyline.
    For multi-point lines, keep as-is.
    """
    pts = np.array(points)
    # if len(pts) <= 2:
    #     t = np.linspace(0, 1, num_interp)
    #     xs = pts[0, 0] + t * (pts[-1, 0] - pts[0, 0])
    #     ys = pts[0, 1] + t * (pts[-1, 1] - pts[0, 1])
    #     return np.column_stack([xs, ys])
    return pts


def spline_at_anchors(points, row_anchors):
    """
    Piecewise-linear interpolation of lane x-positions at row anchor y-values.
    Connects consecutive annotation points with straight lines and samples
    x at each anchor.  Returns array of shape (num_anchors, 2) with [x, y].
    """
    pts = np.array(points)
    xs, ys = pts[:, 0], pts[:, 1]

    y_min, y_max = ys.min(), ys.max()
    if len(xs) < 2:
        result = np.full((len(row_anchors), 2), -99999.0)
        result[:, 1] = row_anchors
        return result

    order = np.argsort(ys)
    ys_sorted = ys[order]
    xs_sorted = xs[order]

    new_x = np.interp(row_anchors, ys_sorted, xs_sorted, left=-99999, right=-99999)

    valid = (row_anchors >= y_min) & (row_anchors <= y_max)
    result = np.column_stack([new_x, row_anchors])
    result[~valid, 0] = -99999

    return result

def draw_lane_on_mask(mask, points, lane_idx, img_width: int):
    """Draw a lane polyline on the segmentation mask with pixel value = lane_idx."""
    pts = interpolate_points(points)
    thickness = lane_thickness_for_points(pts, img_width=img_width, base=LANE_DRAW_WIDTH)
    for i in range(len(pts) - 1):
        pt0 = (int(pts[i, 0]), int(pts[i, 1]))
        pt1 = (int(pts[i + 1, 0]), int(pts[i + 1, 1]))
        cv2.line(mask, pt0, pt1, (lane_idx,), thickness=thickness)


def draw_stop_line_on_mask(mask, points, img_width: int) -> None:
    """
    Draw stop lines on the seg mask without using calc_k / slot assignment.
    Horizontal polylines are skipped by k_neg/k_pos (theta≈0); this path always draws them.
    """
    pts = interpolate_points(points)
    if len(pts) < 2:
        return
    thickness = max(STOP_LINE_DRAW_THICKNESS, LANE_DRAW_WIDTH)
    for i in range(len(pts) - 1):
        pt0 = (int(pts[i, 0]), int(pts[i, 1]))
        pt1 = (int(pts[i + 1, 0]), int(pts[i + 1, 1]))
        cv2.line(mask, pt0, pt1, (STOP_LINE_MASK_VALUE,), thickness=thickness)

def draw_debug_image(
    img,
    bin_label,
    all_points,
    row_anchors,
    slot_raw_pts=None,
    slot_labels=None,
    shapes=None,
):
    """
    Draw lanes on a copy of the image. Line color follows the lane class label string.
    Overlay text is ``slot:class_name``. If ``shapes`` is set, ``stop_line`` polylines
    are drawn too (they are not assigned to slots).
    """
    vis = img.copy()
    slot_labels = slot_labels or {}

    for slot in range(len(bin_label)):
        if not bin_label[slot]:
            continue
        pts = all_points[slot]
        valid = pts[:, 0] > -99990
        if not np.any(valid):
            continue

        class_label = slot_labels.get(slot)
        color = bgr_color_for_lane_label(class_label) if class_label else (180, 180, 180)
        valid_pts = pts[valid].astype(np.int32)

        for k in range(len(valid_pts) - 1):
            pt0 = tuple(valid_pts[k])
            pt1 = tuple(valid_pts[k + 1])
            cv2.line(vis, pt0, pt1, color, thickness=3)

        if slot_raw_pts and slot in slot_raw_pts:
            for rx, ry in slot_raw_pts[slot]:
                center = (int(rx), int(ry))
                cv2.circle(vis, center, 7, (255, 255, 255), -1)
                cv2.circle(vis, center, 7, color, 2)

        mid_idx = len(valid_pts) // 2
        tx, ty = int(valid_pts[mid_idx, 0]), int(valid_pts[mid_idx, 1])
        label = _debug_overlay_text(slot, class_label)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(vis, (tx - 2, ty - th - 4), (tx + tw + 2, ty + 4), (0, 0, 0), -1)
        cv2.putText(vis, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    if shapes:
        for shape in shapes:
            if not isinstance(shape, dict):
                continue
            if shape.get("label", "") in SKIP_LABELS:
                continue
            if not is_stop_line_shape(shape):
                continue
            pts = shape.get("points", [])
            if len(pts) < 2:
                continue
            lbl = str(shape.get("label", "")).strip()
            color = bgr_color_for_lane_label(lbl)
            arr = np.asarray(pts, dtype=np.float32)
            for i in range(len(arr) - 1):
                p0 = (int(arr[i, 0]), int(arr[i, 1]))
                p1 = (int(arr[i + 1, 0]), int(arr[i + 1, 1]))
                cv2.line(vis, p0, p1, color, thickness=4)
            mx, my = int(np.mean(arr[:, 0])), int(np.mean(arr[:, 1]))
            t = _debug_overlay_text(None, lbl)
            (tw, th), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(vis, (mx - 2, my - th - 4), (mx + tw + 2, my + 4), (0, 0, 0), -1)
            cv2.putText(vis, t, (mx, my), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    return vis


def flatten_points_for_calc_k(points):
    """Return lane points in the original annotation order."""
    return list(points)


def process_one_image(shapes, img_height, img_width, row_anchors):
    """
    Process one image's annotations.
    Returns: (bin_label, seg_mask, all_points, slot_raw_pts, num_valid_lanes, slot_labels)
      slot_raw_pts: dict mapping slot index -> list of [x,y] annotation points
      slot_labels: dict slot index -> exact JSON class label string for debug visualization
    """

    # Filter out lanes really close to curb lane lines.
    shapes = filter_boundary_close_lanes_keep_curb(shapes, img_width=img_width)

    # Curvelane slot logic uses signed lane angle; horizontal stop lines have theta≈0 and
    # never land in k_neg/k_pos. Process stop_line shapes separately (mask + JSON only).
    lanes = []
    lane_labels = []
    for shape in shapes:
        if not isinstance(shape, dict):
            continue
        label = str(shape.get("label", "")).strip()
        if label in SKIP_LABELS:
            continue
        if is_stop_line_shape(shape):
            continue
        pts = shape.get("points", [])
        if len(pts) < 2:
            continue
        sorted_pts = flatten_points_for_calc_k(pts)
        lanes.append(sorted_pts)
        lane_labels.append(label)


    if not lanes:
        bin_label = [0] * NUM_LANES
        seg_mask = np.zeros((img_height, img_width), dtype=np.uint8)
        all_points = np.full((NUM_LANES, len(row_anchors), 2), -99999.0)
        all_points[:, :, 1] = np.tile(row_anchors, (NUM_LANES, 1))
        for shape in shapes:
            if not isinstance(shape, dict):
                continue
            if shape.get("label", "") in SKIP_LABELS:
                continue
            if not is_stop_line_shape(shape):
                continue
            pts = shape.get("points", [])
            if len(pts) < 2:
                continue
            draw_stop_line_on_mask(seg_mask, pts, img_width=img_width)
        return bin_label, seg_mask, all_points, {}, 0, {}

    ks = np.array([calc_k(lane, img_height, img_width) for lane in lanes])
    ks_theta = np.array([calc_k(lane, img_height, img_width, angle=True) for lane in lanes])

    k_neg = ks[ks_theta < 0].copy()
    k_neg_theta = ks_theta[ks_theta < 0].copy()
    k_pos = ks[ks_theta > 0].copy()
    k_pos_theta = ks_theta[ks_theta > 0].copy()

    k_neg = k_neg[k_neg_theta != -10]
    k_pos = k_pos[k_pos_theta != -10]
    k_neg.sort()
    k_pos.sort()

    seg_mask = np.zeros((img_height, img_width), dtype=np.uint8)
    bin_label = [0] * NUM_LANES
    all_points = np.full((NUM_LANES, len(row_anchors), 2), -99999.0)
    all_points[:, :, 1] = np.tile(row_anchors, (NUM_LANES, 1))
    slot_raw_pts = {}
    slot_labels = {}

    num_valid = 0

    for idx in range(min(len(k_neg), HALF_LANES)):
        matches = np.where(ks == k_neg[idx])[0]
        if len(matches) == 0:
            continue
        which_lane = matches[0]
        slot = HALF_LANES - 1 - idx
        draw_lane_on_mask(seg_mask, lanes[which_lane], slot + 1, img_width=img_width)
        bin_label[slot] = 1
        all_points[slot] = spline_at_anchors(lanes[which_lane], row_anchors)
        slot_raw_pts[slot] = lanes[which_lane]
        slot_labels[slot] = lane_labels[which_lane]
        num_valid += 1

    for idx in range(min(len(k_pos), HALF_LANES)):
        matches = np.where(ks == k_pos[-(idx + 1)])[0]
        if len(matches) == 0:
            continue
        which_lane = matches[0]
        slot = HALF_LANES + idx
        draw_lane_on_mask(seg_mask, lanes[which_lane], slot + 1, img_width=img_width)
        bin_label[slot] = 1
        all_points[slot] = spline_at_anchors(lanes[which_lane], row_anchors)
        slot_raw_pts[slot] = lanes[which_lane]
        slot_labels[slot] = lane_labels[which_lane]
        num_valid += 1

    # Explicitly add STOP_LINE shapes to the mask (because they won't be assigned to any slot)
    for shape in shapes:
        if not isinstance(shape, dict):
            continue
        if shape.get("label", "") in SKIP_LABELS:
            continue
        if not is_stop_line_shape(shape):
            continue
        pts = shape.get("points", [])
        if len(pts) < 2:
            continue
        draw_stop_line_on_mask(seg_mask, pts, img_width=img_width)

    return bin_label, seg_mask, all_points, slot_raw_pts, num_valid, slot_labels


def shapes_to_curvelanes_lines(shapes, img_width: int):
    """
    Convert LabelMe shapes to CurveLanes-like label payload:
    {
      "Lines": [[{"x":..,"y":..}, ...], ...],
      "LineLabels": [...],          # original labelme label
      "LineClassIds": [...],        # from ACTIVE_LANE_CLASS_TO_ID
      "LaneClassMap": {name: id}
    }
    """
    class_name_by_shape_idx = {}
    if CLASS_MODE == 3:
        class_name_by_shape_idx = assign_three_class_names(shapes, img_width=img_width)
    lines = []
    line_labels = []
    line_class_ids = []
    for idx, shape in enumerate(shapes):
        label = shape.get("label", "")
        if label in SKIP_LABELS:
            continue
        pts = shape.get("points", [])
        if len(pts) < 2:
            continue
        line = []
        for p in pts:
            if len(p) < 2:
                continue
            line.append({"x": float(p[0]), "y": float(p[1])})
        if len(line) >= 2:
            lines.append(line)
            line_labels.append(str(label))
            if CLASS_MODE in (8, 9):
                key = str(label).strip()
                fallback = ACTIVE_LANE_CLASS_TO_ID.get("invisible_line", 0)
                cls_id = int(ACTIVE_LANE_CLASS_TO_ID.get(key, fallback))
            else:
                key = class_name_by_shape_idx.get(idx, "other_lanes")
                cls_id = int(ACTIVE_LANE_CLASS_TO_ID.get(key, ACTIVE_LANE_CLASS_TO_ID["other_lanes"]))
            line_class_ids.append(cls_id)

    return {
        "Lines": lines,
        "LineLabels": line_labels,
        "LineClassIds": line_class_ids,
        "LaneClassMap": dict(ACTIVE_LANE_CLASS_TO_ID),
    }


def split_sessions_by_ratio(sessions, train_ratio, valid_ratio, seed=42):
    """Split whole sessions into train/valid/test sets."""
    if train_ratio <= 0 or valid_ratio < 0 or train_ratio + valid_ratio >= 1:
        raise ValueError("Need train_ratio > 0, valid_ratio >= 0, and train_ratio + valid_ratio < 1.")

    rng = np.random.default_rng(seed)
    sessions = list(sessions)
    rng.shuffle(sessions)

    n_total = len(sessions)
    n_train = int(n_total * train_ratio)
    n_valid = int(n_total * valid_ratio)

    train_sessions = sessions[:n_train]
    valid_sessions = sessions[n_train:n_train + n_valid]
    test_sessions = sessions[n_train + n_valid:]
    return train_sessions, valid_sessions, test_sessions


def discover_session_folders(input_dir):
    """Recursively find session directories that contain both images/ and Annotations/."""
    sessions = []
    input_dir = os.path.abspath(input_dir)

    for root, dirs, _files in os.walk(input_dir):
        if {"images", "labels"}.issubset(set(dirs)):
            sessions.append(os.path.relpath(root, input_dir))

    return sorted(sessions)


def ensure_split_dirs(output_dir, split_name):
    split_root = os.path.join(output_dir, split_name)
    os.makedirs(os.path.join(split_root, "images"), exist_ok=True)
    os.makedirs(os.path.join(split_root, "labels"), exist_ok=True)
    os.makedirs(os.path.join(split_root, "debug_vis"), exist_ok=True)
    return split_root


def process_split_sessions(input_dir, output_dir, split_name, sessions, row_anchors):
    """
    Process a list of sessions and write split-specific files.
    Returns split entries used to write list/cache files plus filtering stats.
    """
    split_root = ensure_split_dirs(output_dir, split_name)
    entries = []
    stats = Counter()

    for session in sessions:
        session_path = os.path.join(input_dir, session)
        json_dir = os.path.join(session_path, "labels")
        images_dir = os.path.join(session_path, "images")

        if not os.path.isdir(json_dir) or not os.path.isdir(images_dir):
            print(f"[{split_name}] Skipping {session}: missing labels or images folder")
            continue

        json_files = sorted(glob.glob(os.path.join(json_dir, "*.json")))
        print(f"[{split_name}] Processing {session}: {len(json_files)} annotations")

        for jf in json_files:
            stats["json_seen"] += 1
            basename = os.path.splitext(os.path.basename(jf))[0]

            img_path = os.path.join(images_dir, basename + ".jpg")
            if not os.path.exists(img_path):
                img_path = os.path.join(images_dir, basename + ".png")
            if not os.path.exists(img_path):
                stats["missing_image"] += 1
                continue

            with open(jf) as f:
                data = json.load(f)

            img_h = data.get("imageHeight", IMG_HEIGHT)
            img_w = data.get("imageWidth", IMG_WIDTH)
            shapes = data.get("shapes", [])
            if not shapes:
                stats["empty_shapes"] += 1
                continue

            raw_out_name = f"{session}_{basename}"
            out_name = re.sub(r"[^A-Za-z0-9._-]", "_", raw_out_name)

            split_img_rel = f"{split_name}/images/{out_name}.jpg"
            split_label_rel = f"{split_name}/labels/{out_name}.lines.json"
            img_rel = f"images/{out_name}.jpg"
            label_rel = f"labels/{out_name}.lines.json"

            # Filter out images that have less than 2 lanes
            if len(shapes) < 2:
                stats["lt_2_raw_shapes"] += 1
                continue

            bin_label, seg_mask, points, slot_raw_pts, num_valid, slot_labels = process_one_image(
                shapes, img_h, img_w, row_anchors
            )
            if sum(bin_label) < 2:
                stats["lt_2_valid_slots_after_assignment"] += 1
                continue
            lines_payload = shapes_to_curvelanes_lines(shapes, img_width=img_w)

            img = cv2.imread(img_path)
            cv2.imwrite(os.path.join(output_dir, split_img_rel), img)
            with open(os.path.join(output_dir, split_label_rel), "w") as f:
                json.dump(lines_payload, f)

            debug_img = draw_debug_image(
                img,
                bin_label,
                points,
                row_anchors,
                slot_raw_pts,
                slot_labels=slot_labels,
                shapes=shapes,
            )
            cv2.imwrite(os.path.join(split_root, f"debug_vis/{out_name}.jpg"), debug_img)

            entries.append({
                "img_rel": img_rel,
                "label_rel": label_rel,
                "split_img_rel": split_img_rel,
                "split_label_rel": split_label_rel,
                "bin_label": bin_label,
                "points": points.tolist(),
                "num_valid": num_valid,
            })
            stats["written"] += 1

    return entries, stats


def collect_valid_samples_from_sessions(input_dir, sessions, row_anchors):
    """
    Collect all valid samples from sessions without writing split files.
    Returns:
      - records: per-sample processed payloads for later split assignment/writing
      - stats: filtering stats equivalent to process_split_sessions pre-write stage
    """
    records = []
    stats = Counter()

    for session in sessions:
        session_path = os.path.join(input_dir, session)
        json_dir = os.path.join(session_path, "labels")
        images_dir = os.path.join(session_path, "images")

        if not os.path.isdir(json_dir) or not os.path.isdir(images_dir):
            print(f"[collect] Skipping {session}: missing labels or images folder")
            continue

        json_files = sorted(glob.glob(os.path.join(json_dir, "*.json")))
        print(f"[collect] Processing {session}: {len(json_files)} annotations")

        for jf in json_files:
            stats["json_seen"] += 1
            basename = os.path.splitext(os.path.basename(jf))[0]

            img_path = os.path.join(images_dir, basename + ".jpg")
            if not os.path.exists(img_path):
                img_path = os.path.join(images_dir, basename + ".png")
            if not os.path.exists(img_path):
                stats["missing_image"] += 1
                continue

            with open(jf) as f:
                data = json.load(f)

            img_h = data.get("imageHeight", IMG_HEIGHT)
            img_w = data.get("imageWidth", IMG_WIDTH)
            shapes = data.get("shapes", [])
            if not shapes:
                stats["empty_shapes"] += 1
                continue

            # # Filter out images that have less than 2 lanes
            # if len(shapes) < 2:
            #     stats["lt_2_raw_shapes"] += 1
            #     continue

            bin_label, seg_mask, points, slot_raw_pts, num_valid, slot_labels = process_one_image(
                shapes, img_h, img_w, row_anchors
            )
            # if sum(bin_label) < 2:
            #     stats["lt_2_valid_slots_after_assignment"] += 1
            #     continue

            lines_payload = shapes_to_curvelanes_lines(shapes, img_width=img_w)
            class_ids = sorted(set(int(cid) for cid in lines_payload.get("LineClassIds", [])))

            raw_out_name = f"{session}_{basename}"
            out_name = re.sub(r"[^A-Za-z0-9._-]", "_", raw_out_name)
            records.append({
                "session": session,
                "basename": basename,
                "out_name": out_name,
                "img_path": img_path,
                "bin_label": bin_label,
                "points": points,
                "slot_raw_pts": slot_raw_pts,
                "num_valid": num_valid,
                "slot_labels": slot_labels,
                "shapes": shapes,
                "lines_payload": lines_payload,
                "class_ids": class_ids,
            })

    return records, stats


def split_records_stratified_by_class(records, train_ratio, valid_ratio, seed=42):
    """
    Stratified split by lane class presence (multi-label).
    Uses a greedy assignment to balance per-class presence counts across splits.
    """
    if train_ratio <= 0 or valid_ratio < 0 or train_ratio + valid_ratio >= 1:
        raise ValueError("Need train_ratio > 0, valid_ratio >= 0, and train_ratio + valid_ratio < 1.")

    n_total = len(records)
    if n_total == 0:
        return [], [], []

    n_train = int(n_total * train_ratio)
    n_valid = int(n_total * valid_ratio)
    n_test = n_total - n_train - n_valid
    split_caps = {"train": n_train, "valid": n_valid, "test": n_test}
    split_names = ["train", "valid", "test"]

    per_class_total = Counter()
    per_record_class_sets = []
    for rec in records:
        cls_set = set(rec.get("class_ids", []))
        per_record_class_sets.append(cls_set)
        for cid in cls_set:
            per_class_total[cid] += 1

    target_class_counts = {
        "train": {cid: per_class_total[cid] * train_ratio for cid in per_class_total},
        "valid": {cid: per_class_total[cid] * valid_ratio for cid in per_class_total},
        "test": {cid: per_class_total[cid] * (1.0 - train_ratio - valid_ratio) for cid in per_class_total},
    }

    rng = np.random.default_rng(seed)
    idxs = list(range(n_total))
    rng.shuffle(idxs)

    def rarity_score(idx):
        cls_set = per_record_class_sets[idx]
        if not cls_set:
            return float("inf")
        return min(per_class_total[cid] for cid in cls_set)

    idxs.sort(key=lambda idx: (rarity_score(idx), -len(per_record_class_sets[idx])))

    split_records = {"train": [], "valid": [], "test": []}
    current_class_counts = {name: Counter() for name in split_names}
    current_sizes = {name: 0 for name in split_names}

    for idx in idxs:
        cls_set = per_record_class_sets[idx]
        best_split = None
        best_score = None

        for split_name in split_names:
            if current_sizes[split_name] >= split_caps[split_name]:
                continue

            score = 0.0
            for cid in cls_set:
                cur = current_class_counts[split_name][cid]
                tgt = target_class_counts[split_name].get(cid, 0.0)
                score += abs((cur + 1) - tgt) - abs(cur - tgt)

            after_size = current_sizes[split_name] + 1
            score += 0.01 * abs(after_size - split_caps[split_name])

            if best_score is None or score < best_score:
                best_score = score
                best_split = split_name

        if best_split is None:
            remaining = [s for s in split_names if current_sizes[s] < split_caps[s]]
            best_split = remaining[0] if remaining else "test"

        split_records[best_split].append(records[idx])
        current_sizes[best_split] += 1
        for cid in cls_set:
            current_class_counts[best_split][cid] += 1

    return split_records["train"], split_records["valid"], split_records["test"]


def write_records_to_split(output_dir, split_name, records, row_anchors):
    """Write pre-collected records into one split folder."""
    split_root = ensure_split_dirs(output_dir, split_name)
    entries = []
    stats = Counter()

    for rec in records:
        out_name = rec["out_name"]
        split_img_rel = f"{split_name}/images/{out_name}.jpg"
        split_label_rel = f"{split_name}/labels/{out_name}.lines.json"
        img_rel = f"images/{out_name}.jpg"
        label_rel = f"labels/{out_name}.lines.json"

        img = cv2.imread(rec["img_path"])
        if img is None:
            stats["missing_image"] += 1
            continue

        cv2.imwrite(os.path.join(output_dir, split_img_rel), img)
        with open(os.path.join(output_dir, split_label_rel), "w") as f:
            json.dump(rec["lines_payload"], f)

        debug_img = draw_debug_image(
            img,
            rec["bin_label"],
            rec["points"],
            row_anchors,
            rec["slot_raw_pts"],
            slot_labels=rec["slot_labels"],
            shapes=rec["shapes"],
        )
        cv2.imwrite(os.path.join(split_root, f"debug_vis/{out_name}.jpg"), debug_img)

        entries.append({
            "img_rel": img_rel,
            "label_rel": label_rel,
            "split_img_rel": split_img_rel,
            "split_label_rel": split_label_rel,
            "bin_label": rec["bin_label"],
            "points": rec["points"].tolist(),
            "num_valid": rec["num_valid"],
        })
        stats["written"] += 1

    return entries, stats


def write_split_files(output_dir, split_name, entries):
    split_root = os.path.join(output_dir, split_name)

    # Requested split image lists
    list_filename = "train.txt" if split_name == "train" else f"{split_name}.txt"

    if split_name == "train":
        with open(os.path.join(split_root, list_filename), "w") as f:
            for entry in entries:
                f.write(f"{entry['img_rel']} \n")
    elif split_name == "valid":
        with open(os.path.join(split_root, list_filename), "w") as f:
            for entry in entries:
                f.write(f"{entry['img_rel']} \n")
    else:
        with open(os.path.join(split_root, list_filename), "w") as f:
            for entry in entries:
                f.write(f"{entry['img_rel']} \n")


def write_cache_file(output_dir, split_name, entries):
    """
    Write split-specific cache json from freshly computed `points`.
    This prevents stale cache files from previous conversions.
    """
    if split_name == "train":
        cache_name = "curvelanes_anno_cache.json"
        key_field = "split_img_rel"  # e.g. train/images/xxx.jpg
    elif split_name == "valid":
        cache_name = "curvelanes_anno_cache.json"
        key_field = "img_rel"  # e.g. images/xxx.jpg
    elif split_name == "test":
        cache_name = "curvelanes_anno_cache.json"
        key_field = "img_rel"
    else:
        raise ValueError(f"Unknown split: {split_name}")

    cache_path = os.path.join(output_dir, split_name, cache_name)
    payload = {}
    for entry in entries:
        payload[entry[key_field]] = entry["points"]

    with open(cache_path, "w") as f:
        json.dump(payload, f)

    print(f"  Wrote cache: {cache_path} ({len(payload)} entries)")

def main():
    parser = argparse.ArgumentParser(description="Convert LabelMe annotations to UFLDv2 format")
    parser.add_argument("--input-dir", required=True,
                        help="Root directory containing session folders anywhere under it (each with images/ and Annotations/)")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for UFLDv2-formatted data")
    parser.add_argument("--num-lanes", type=int, default=6)
    parser.add_argument("--lane-width", type=int, default=16)
    parser.add_argument("--train-ratio", type=float, default=0.8,
                        help="Train split ratio by session directories")
    parser.add_argument("--valid-ratio", type=float, default=0.1,
                        help="Validation split ratio by session directories")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for directory-level split")
    parser.add_argument("--split-mode", type=str, choices=["session", "stratified"], default="session",
                        help="Split strategy: session=legacy whole-session split, stratified=class-balanced image-level split")
    parser.add_argument("--include-stop-lines", action="store_true", default=False,
                        help="Include stop_line annotations")
    parser.add_argument("--class-mode", type=int, choices=[3, 9], default=9,
                        help="Lane class mode: 9=full lane taxonomy (default), 3=egoleft/egoright/other_lanes")
    args = parser.parse_args()

    global NUM_LANES, HALF_LANES, LANE_DRAW_WIDTH, SKIP_LABELS
    global CLASS_MODE, ACTIVE_LANE_CLASSES, ACTIVE_LANE_CLASS_TO_ID
    NUM_LANES = args.num_lanes
    HALF_LANES = NUM_LANES // 2
    LANE_DRAW_WIDTH = args.lane_width
    CLASS_MODE = args.class_mode
    ACTIVE_LANE_CLASSES, ACTIVE_LANE_CLASS_TO_ID = build_lane_taxonomy(CLASS_MODE)
    if not(args.include_stop_lines):
        SKIP_LABELS = {"stop_line"}
    else:
        SKIP_LABELS = set()

    row_anchors = np.array(list(range(ROW_ANCHOR_START, ROW_ANCHOR_END, ROW_ANCHOR_STEP)))

    sessions = discover_session_folders(args.input_dir)

    if args.split_mode == "session":
        train_sessions, valid_sessions, test_sessions = split_sessions_by_ratio(
            sessions,
            train_ratio=args.train_ratio,
            valid_ratio=args.valid_ratio,
            seed=args.seed
        )

        print("\nSession split (directory-level):")
        print(f"  Train sessions: {len(train_sessions)}")
        print(f"  Valid sessions: {len(valid_sessions)}")
        print(f"  Test sessions : {len(test_sessions)}")

        train_entries, train_stats = process_split_sessions(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            split_name="train",
            sessions=train_sessions,
            row_anchors=row_anchors
        )
        valid_entries, valid_stats = process_split_sessions(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            split_name="valid",
            sessions=valid_sessions,
            row_anchors=row_anchors
        )
        test_entries, test_stats = process_split_sessions(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            split_name="test",
            sessions=test_sessions,
            row_anchors=row_anchors
        )
        all_stats = train_stats + valid_stats + test_stats
    else:
        print("\nStratified split (image-level, class-balanced by presence):")
        all_records, base_stats = collect_valid_samples_from_sessions(
            input_dir=args.input_dir,
            sessions=sessions,
            row_anchors=row_anchors,
        )
        train_records, valid_records, test_records = split_records_stratified_by_class(
            all_records,
            train_ratio=args.train_ratio,
            valid_ratio=args.valid_ratio,
            seed=args.seed,
        )
        print(f"  Train records: {len(train_records)}")
        print(f"  Valid records: {len(valid_records)}")
        print(f"  Test records : {len(test_records)}")

        train_entries, train_write_stats = write_records_to_split(
            args.output_dir, "train", train_records, row_anchors
        )
        valid_entries, valid_write_stats = write_records_to_split(
            args.output_dir, "valid", valid_records, row_anchors
        )
        test_entries, test_write_stats = write_records_to_split(
            args.output_dir, "test", test_records, row_anchors
        )
        all_stats = base_stats + train_write_stats + valid_write_stats + test_write_stats

    write_split_files(args.output_dir, "train", train_entries)
    write_split_files(args.output_dir, "valid", valid_entries)
    write_split_files(args.output_dir, "test", test_entries)
    # write_cache_file(args.output_dir, "train", train_entries)
    # write_cache_file(args.output_dir, "valid", valid_entries)
    # write_cache_file(args.output_dir, "test", test_entries)

    all_entries = train_entries + valid_entries + test_entries
    lane_counts = Counter()
    for e in all_entries:
        lane_counts[sum(e["bin_label"])] += 1

    print(f"\n{'='*50}")
    print(f"Conversion complete!")
    print(f"  Total images: {len(all_entries)}")
    print(f"  Train: {len(train_entries)}, Valid: {len(valid_entries)}, Test: {len(test_entries)}")
    print(f"  Num lanes (slots): {NUM_LANES}")
    print(f"  Lane class mode: {CLASS_MODE} ({len(ACTIVE_LANE_CLASSES)} classes)")
    print(f"  Row anchors: {len(row_anchors)} (y={ROW_ANCHOR_START} to y={ROW_ANCHOR_END}, step={ROW_ANCHOR_STEP})")
    print(f"  Stop lines: {'included' if not SKIP_LABELS else 'skipped'}")
    print(f"\n  Original annotations seen: {all_stats.get('json_seen', 0)}")
    print(f"  Samples written         : {all_stats.get('written', 0)}")
    print(f"  Samples filtered out    : {all_stats.get('json_seen', 0) - all_stats.get('written', 0)}")
    print(f"    Missing image file              : {all_stats.get('missing_image', 0)}")
    print(f"    Empty shapes list               : {all_stats.get('empty_shapes', 0)}")
    print(f"    Fewer than 2 raw shapes         : {all_stats.get('lt_2_raw_shapes', 0)}")
    print(f"    Fewer than 2 valid slots after assignment: {all_stats.get('lt_2_valid_slots_after_assignment', 0)}")
    print(f"\n  Lane count distribution (after slot assignment):")
    for k in sorted(lane_counts.keys()):
        print(f"    {k} lanes: {lane_counts[k]} images")


if __name__ == "__main__":
    main()
