import os
import re
import cv2
import json
import glob
import warnings
import argparse
import numpy as np
from collections import Counter

try:
    warnings.simplefilter("ignore", np.exceptions.RankWarning)
except AttributeError:
    warnings.simplefilter("ignore", np.RankWarning)

NUM_LANES = 10
HALF_LANES = NUM_LANES // 2
IMG_WIDTH = 1920
IMG_HEIGHT = 1080
LANE_DRAW_WIDTH = 16
MIN_LANE_LENGTH = 80
ROW_ANCHOR_START = 400 # 64 row anchors
ROW_ANCHOR_END = 1080
ROW_ANCHOR_STEP = 10
DEDUP_THRESHOLD = 30
MERGE_Y_GAP = 100
MERGE_X_DIFF = 80
SKIP_LABELS = {"stop_line"}


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


def merge_collinear_lanes(lanes, merge_y_gap=None, merge_x_diff=None):
    """
    Merge lane segments that are collinear continuations of each other
    (e.g. a dashed-white segment followed by a solid-white segment on the
    same physical lane).  Two segments are merged when:
      1. The vertical gap between them is <= merge_y_gap pixels.
      2. Their extrapolated x-positions at the meeting point differ by
         <= merge_x_diff pixels.
    Runs iteratively until no more merges are possible.
    """
    if merge_y_gap is None:
        merge_y_gap = MERGE_Y_GAP
    if merge_x_diff is None:
        merge_x_diff = MERGE_X_DIFF
    if len(lanes) <= 1:
        return lanes

    fits = [_lane_fit(l) for l in lanes]

    changed = True
    while changed:
        changed = False
        n = len(lanes)
        for i in range(n):
            if fits[i] is None:
                continue
            for j in range(i + 1, n):
                if fits[j] is None:
                    continue
                y_min_i, y_max_i, curve_i = fits[i]
                y_min_j, y_max_j, curve_j = fits[j]

                if y_min_i <= y_min_j:
                    upper_max, lower_min = y_max_i, y_min_j
                    curve_upper, curve_lower = curve_i, curve_j
                else:
                    upper_max, lower_min = y_max_j, y_min_i
                    curve_upper, curve_lower = curve_j, curve_i

                gap = lower_min - upper_max
                if gap > merge_y_gap:
                    continue

                meeting_y = (upper_max + lower_min) / 2.0
                x_upper = np.polyval(curve_upper, meeting_y)
                x_lower = np.polyval(curve_lower, meeting_y)
                if abs(x_upper - x_lower) > merge_x_diff:
                    continue

                merged_pts = sorted(lanes[i] + lanes[j], key=lambda p: -p[1])
                lanes[i] = merged_pts
                fits[i] = _lane_fit(merged_pts)
                lanes[j] = None
                fits[j] = None
                changed = True
                break
            if changed:
                break

        if changed:
            paired = [(l, f) for l, f in zip(lanes, fits) if l is not None]
            lanes = [p[0] for p in paired]
            fits = [p[1] for p in paired]

    return lanes


def deduplicate_lanes(lanes, row_anchors, threshold=None):
    """
    Remove duplicate lanes that trace the same physical edge (e.g. a curb
    annotation overlapping a line-type annotation).  Two lanes are duplicates
    when their mean x-distance at shared row anchors falls below `threshold`.
    The lane with more annotation points is kept.
    """
    if threshold is None:
        threshold = DEDUP_THRESHOLD
    if len(lanes) <= 1:
        return lanes

    lane_xs = []
    for lane in lanes:
        pts = np.array(lane)
        xs, ys = pts[:, 0], pts[:, 1]
        deg = _fit_degree(len(xs))
        y_min, y_max = ys.min(), ys.max()
        if deg < 1 or (y_max - y_min) < 1:
            lane_xs.append(None)
            continue
        curve = np.polyfit(ys, xs, deg=deg)
        valid = (row_anchors >= y_min) & (row_anchors <= y_max)
        x_at_anchors = np.polyval(curve, row_anchors)
        x_at_anchors[~valid] = np.nan
        lane_xs.append(x_at_anchors)

    n = len(lanes)
    keep = [True] * n
    for i in range(n):
        if not keep[i] or lane_xs[i] is None:
            continue
        for j in range(i + 1, n):
            if not keep[j] or lane_xs[j] is None:
                continue
            both_valid = ~np.isnan(lane_xs[i]) & ~np.isnan(lane_xs[j])
            if not np.any(both_valid):
                continue
            mean_dist = np.mean(np.abs(lane_xs[i][both_valid] - lane_xs[j][both_valid]))
            if mean_dist < threshold:
                if len(lanes[i]) >= len(lanes[j]):
                    keep[j] = False
                else:
                    keep[i] = False
                    break

    return [lane for lane, k in zip(lanes, keep) if k]


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


def interpolate_points(points, num_interp=100):
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

def draw_lane_on_mask(mask, points, lane_idx):
    """Draw a lane polyline on the segmentation mask with pixel value = lane_idx."""
    pts = interpolate_points(points)
    for i in range(len(pts) - 1):
        pt0 = (int(pts[i, 0]), int(pts[i, 1]))
        pt1 = (int(pts[i + 1, 0]), int(pts[i + 1, 1]))
        cv2.line(mask, pt0, pt1, (lane_idx,), thickness=LANE_DRAW_WIDTH)


LANE_COLORS = [
    (255, 0, 0),      # slot 0 — blue
    (0, 255, 0),      # slot 1 — green
    (0, 0, 255),      # slot 2 — red
    (255, 255, 0),    # slot 3 — cyan
    (255, 0, 255),    # slot 4 — magenta
    (0, 255, 255),    # slot 5 — yellow
    (128, 128, 255),  # slot 6+
    (128, 255, 128),
]


def draw_debug_image(img, bin_label, all_points, row_anchors, slot_raw_pts=None):
    """
    Draw processed lanes on a copy of the source image.  Each active slot
    is drawn in a distinct color with the slot index rendered as text near
    the lane midpoint.  If slot_raw_pts is provided, the original annotation
    points are drawn as circles so you can see what the curve was fitted to.
    """
    vis = img.copy()

    for slot in range(len(bin_label)):
        if not bin_label[slot]:
            continue
        pts = all_points[slot]
        valid = pts[:, 0] > -99990
        if not np.any(valid):
            continue

        color = LANE_COLORS[slot % len(LANE_COLORS)]
        valid_pts = pts[valid].astype(np.int32)

        for k in range(len(valid_pts) - 1):
            pt0 = tuple(valid_pts[k])
            pt1 = tuple(valid_pts[k + 1])
            cv2.line(vis, pt0, pt1, color, thickness=3)
            # cv2.circle(vis, pt0, 7, (0, 0, 0), -1)
            # cv2.circle(vis, pt0, 7, color, 2)
            # cv2.circle(vis, pt1, 7, (0, 0, 0), -1)
            # cv2.circle(vis, pt1, 7, color, 2)
            # cv2.putText(vis, str(k), pt0, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        if slot_raw_pts and slot in slot_raw_pts:
            for rx, ry in slot_raw_pts[slot]:
                center = (int(rx), int(ry))
                cv2.circle(vis, center, 7, (255, 255, 255), -1)
                cv2.circle(vis, center, 7, color, 2)

        mid_idx = len(valid_pts) // 2
        tx, ty = int(valid_pts[mid_idx, 0]), int(valid_pts[mid_idx, 1])
        label = str(slot)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
        cv2.rectangle(vis, (tx - 2, ty - th - 4), (tx + tw + 2, ty + 4), (0, 0, 0), -1)
        cv2.putText(vis, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    return vis


def flatten_points_for_calc_k(points):
    """Return lane points in the original annotation order."""
    return list(points)


def process_one_image(shapes, img_height, img_width, row_anchors):
    """
    Process one image's annotations.
    Returns: (bin_label, seg_mask, all_points, slot_raw_pts, num_valid_lanes)
      slot_raw_pts: dict mapping slot index -> list of [x,y] annotation points
    """
    lanes = []
    for shape in shapes:
        label = shape["label"]
        if label in SKIP_LABELS:
            continue
        pts = shape["points"]
        if len(pts) < 2:
            continue
        sorted_pts = flatten_points_for_calc_k(pts)
        lanes.append(sorted_pts)

    # lanes = merge_collinear_lanes(lanes)
    # lanes = deduplicate_lanes(lanes, row_anchors)

    if not lanes:
        bin_label = [0] * NUM_LANES
        seg_mask = np.zeros((img_height, img_width), dtype=np.uint8)
        all_points = np.full((NUM_LANES, len(row_anchors), 2), -99999.0)
        all_points[:, :, 1] = np.tile(row_anchors, (NUM_LANES, 1))
        return bin_label, seg_mask, all_points, {}, 0

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

    num_valid = 0

    for idx in range(min(len(k_neg), HALF_LANES)):
        matches = np.where(ks == k_neg[idx])[0]
        if len(matches) == 0:
            continue
        which_lane = matches[0]
        slot = HALF_LANES - 1 - idx
        draw_lane_on_mask(seg_mask, lanes[which_lane], slot + 1)
        bin_label[slot] = 1
        all_points[slot] = spline_at_anchors(lanes[which_lane], row_anchors)
        slot_raw_pts[slot] = lanes[which_lane]
        num_valid += 1

    for idx in range(min(len(k_pos), HALF_LANES)):
        matches = np.where(ks == k_pos[-(idx + 1)])[0]
        if len(matches) == 0:
            continue
        which_lane = matches[0]
        slot = HALF_LANES + idx
        draw_lane_on_mask(seg_mask, lanes[which_lane], slot + 1)
        bin_label[slot] = 1
        all_points[slot] = spline_at_anchors(lanes[which_lane], row_anchors)
        slot_raw_pts[slot] = lanes[which_lane]
        num_valid += 1

    return bin_label, seg_mask, all_points, slot_raw_pts, num_valid


def shapes_to_curvelanes_lines(shapes):
    """
    Convert LabelMe shapes to CurveLanes-like label payload:
    {"Lines": [[{"x":..,"y":..}, ...], ...]}
    """
    lines = []
    for shape in shapes:
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
    return {"Lines": lines}


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


def ensure_split_dirs(output_dir, split_name):
    split_root = os.path.join(output_dir, split_name)
    os.makedirs(os.path.join(split_root, "images"), exist_ok=True)
    os.makedirs(os.path.join(split_root, "labels"), exist_ok=True)
    # os.makedirs(os.path.join(split_root, "segs"), exist_ok=True)
    # os.makedirs(os.path.join(split_root, "debug_vis"), exist_ok=True)
    return split_root


def process_split_sessions(input_dir, output_dir, split_name, sessions, row_anchors):
    """
    Process a list of sessions and write split-specific files.
    Returns split entries used to write list/cache files.
    """
    split_root = ensure_split_dirs(output_dir, split_name)
    entries = []

    for session in sessions:
        session_path = os.path.join(input_dir, session)
        json_dir = os.path.join(session_path, "labels_json")
        images_dir = os.path.join(session_path, "images")

        if not os.path.isdir(json_dir) or not os.path.isdir(images_dir):
            print(f"[{split_name}] Skipping {session}: missing labels or images folder")
            continue

        json_files = sorted(glob.glob(os.path.join(json_dir, "*.json")))
        print(f"[{split_name}] Processing {session}: {len(json_files)} annotations")

        for jf in json_files:
            basename = os.path.splitext(os.path.basename(jf))[0]

            img_path = os.path.join(images_dir, basename + ".jpg")
            if not os.path.exists(img_path):
                img_path = os.path.join(images_dir, basename + ".png")
            if not os.path.exists(img_path):
                continue

            with open(jf) as f:
                data = json.load(f)

            img_h = data.get("imageHeight", IMG_HEIGHT)
            img_w = data.get("imageWidth", IMG_WIDTH)
            shapes = data.get("shapes", [])
            if not shapes:
                continue

            raw_out_name = f"{session}_{basename}"
            out_name = re.sub(r"[^A-Za-z0-9._-]", "_", raw_out_name)

            split_img_rel = f"{split_name}/images/{out_name}.jpg"
            split_label_rel = f"{split_name}/labels/{out_name}.lines.json"
            img_rel = f"images/{out_name}.jpg"
            label_rel = f"labels/{out_name}.lines.json"
            # split_seg_rel = f"{split_name}/segs/{out_name}.png"
            # local_img_rel = f"images/{out_name}.jpg"
            # local_label_rel = f"labels/{out_name}.lines.json"
            # local_seg_rel = f"segs/{out_name}.png"

            bin_label, seg_mask, points, slot_raw_pts, num_valid = process_one_image(
                shapes, img_h, img_w, row_anchors
            )
            lines_payload = shapes_to_curvelanes_lines(shapes)

            img = cv2.imread(img_path)
            cv2.imwrite(os.path.join(output_dir, split_img_rel), img)
            with open(os.path.join(output_dir, split_label_rel), "w") as f:
                json.dump(lines_payload, f)
            # cv2.imwrite(os.path.join(output_dir, split_seg_rel), seg_mask)

            # debug_img = draw_debug_image(img, bin_label, points, row_anchors, slot_raw_pts)
            # cv2.imwrite(os.path.join(split_root, f"debug_vis/{out_name}.jpg"), debug_img)

            entries.append({
                "img_rel": img_rel,
                "label_rel": label_rel,
                "split_img_rel": split_img_rel,
                "split_label_rel": split_label_rel,
                # "seg_rel": split_seg_rel,
                # "img_local_rel": local_img_rel,
                # "label_local_rel": local_label_rel,
                # "seg_local_rel": local_seg_rel,
                "bin_label": bin_label,
                "points": points.tolist(),
                "num_valid": num_valid,
            })

    return entries


def write_split_files(output_dir, split_name, entries):
    split_root = os.path.join(output_dir, split_name)

    # Requested split image lists
    list_filename = "train.txt" if split_name == "train" else f"{split_name}.txt"

    if split_name == "train":
        with open(os.path.join(split_root, list_filename), "w") as f:
            for entry in entries:
                f.write(f"{entry['split_img_rel']} \n")
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
                        help="Root directory containing session folders (each with images/ and labels/)")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for UFLDv2-formatted data")
    parser.add_argument("--num-lanes", type=int, default=10)
    parser.add_argument("--lane-width", type=int, default=16)
    parser.add_argument("--train-ratio", type=float, default=0.8,
                        help="Train split ratio by session directories")
    parser.add_argument("--valid-ratio", type=float, default=0.1,
                        help="Validation split ratio by session directories")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for directory-level split")
    parser.add_argument("--dedup-threshold", type=float, default=50,
                        help="Max mean x-distance (px) to consider two lanes as duplicates (default: 50)")
    parser.add_argument("--merge-y-gap", type=float, default=50,
                        help="Max vertical gap (px) between two segments to merge them (default: 100)")
    parser.add_argument("--merge-x-diff", type=float, default=30,
                        help="Max x-distance (px) at the meeting point to merge two segments (default: 80)")
    parser.add_argument("--skip-stop-lines", action="store_true", default=True,
                        help="Skip stop_line annotations (default: True)")
    parser.add_argument("--include-stop-lines", action="store_true", default=False,
                        help="Include stop_line annotations")
    args = parser.parse_args()

    global NUM_LANES, HALF_LANES, LANE_DRAW_WIDTH, SKIP_LABELS, DEDUP_THRESHOLD, MERGE_Y_GAP, MERGE_X_DIFF
    NUM_LANES = args.num_lanes
    HALF_LANES = NUM_LANES // 2
    LANE_DRAW_WIDTH = args.lane_width
    DEDUP_THRESHOLD = args.dedup_threshold
    MERGE_Y_GAP = args.merge_y_gap
    MERGE_X_DIFF = args.merge_x_diff
    if args.skip_stop_lines or not(args.include_stop_lines):
        SKIP_LABELS = {"stop_line"}
    else:
        SKIP_LABELS = set()

    row_anchors = np.array(list(range(ROW_ANCHOR_START, ROW_ANCHOR_END, ROW_ANCHOR_STEP)))

    sessions = sorted([
        d for d in os.listdir(args.input_dir)
        if os.path.isdir(os.path.join(args.input_dir, d))
           and os.path.isdir(os.path.join(args.input_dir, d, "images"))
           and os.path.isdir(os.path.join(args.input_dir, d, "labels_json"))
    ])

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

    train_entries = process_split_sessions(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        split_name="train",
        sessions=train_sessions,
        row_anchors=row_anchors
    )
    valid_entries = process_split_sessions(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        split_name="valid",
        sessions=valid_sessions,
        row_anchors=row_anchors
    )
    test_entries = process_split_sessions(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        split_name="test",
        sessions=test_sessions,
        row_anchors=row_anchors
    )

    write_split_files(args.output_dir, "train", train_entries)
    write_split_files(args.output_dir, "valid", valid_entries)
    write_split_files(args.output_dir, "test", test_entries)
    write_cache_file(args.output_dir, "train", train_entries)
    write_cache_file(args.output_dir, "valid", valid_entries)
    write_cache_file(args.output_dir, "test", test_entries)

    all_entries = train_entries + valid_entries + test_entries
    lane_counts = Counter()
    for e in all_entries:
        lane_counts[sum(e["bin_label"])] += 1

    print(f"\n{'='*50}")
    print(f"Conversion complete!")
    print(f"  Total images: {len(all_entries)}")
    print(f"  Train: {len(train_entries)}, Valid: {len(valid_entries)}, Test: {len(test_entries)}")
    print(f"  Num lanes (slots): {NUM_LANES}")
    print(f"  Row anchors: {len(row_anchors)} (y={ROW_ANCHOR_START} to y={ROW_ANCHOR_END}, step={ROW_ANCHOR_STEP})")
    print(f"  Stop lines: {'included' if not SKIP_LABELS else 'skipped'}")
    print(f"\n  Lane count distribution (after slot assignment):")
    for k in sorted(lane_counts.keys()):
        print(f"    {k} lanes: {lane_counts[k]} images")
    print(f"\n  Output: {args.output_dir}")
    print(f"    train/train_gt.txt ({len(train_entries)} entries)")
    print(f"    train/train.txt")
    print(f"    train/curvelanes_anno_cache.json")
    print(f"    valid/valid_gt.txt ({len(valid_entries)} entries)")
    print(f"    valid/valid.txt")
    print(f"    valid/culane_anno_cache_val.json")
    print(f"    valid/valid_for_culane_style.txt ({len(valid_entries)} entries)")
    print(f"    test/test.txt ({len(test_entries)} entries)")
    print(f"    test/culane_anno_cache_test.json")


if __name__ == "__main__":
    main()
