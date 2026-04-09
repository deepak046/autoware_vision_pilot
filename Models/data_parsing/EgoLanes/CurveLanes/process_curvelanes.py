#! /usr/bin/env python3

import argparse
import json
import os
import shutil
import warnings
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw


# ============================= Format functions ============================= #

# Lane taxonomies (match convert_labelme_to_curvelanes .lines.json).
LANE_CLASSES_8 = [
    "continuous_white_line",
    "continuous_yellow_line",
    "dashed_white_line",
    "double_white_lines",
    "double_yellow_lines",
    "curb_line",
    "stop_line",
    "invisible_line",
]
LANE_CLASSES_3 = [
    "egoleft_lane",
    "egoright_lane",
    "other",
]

# RGB colors for semantic id 1..N (background 0 = black)
LANE_CLASS_SEMANTIC_RGB_8 = [
    (240, 240, 240),
    (0, 255, 255),
    (200, 200, 200),
    (220, 220, 220),
    (0, 220, 255),
    (0, 140, 255),
    (0, 0, 255),
    (128, 128, 128),
]
LANE_CLASS_SEMANTIC_RGB_3 = [
    (0, 255, 0),      # egoleft_lane
    (255, 0, 0),      # egoright_lane
    (180, 180, 180),  # other
]

CLASS_MODE = 8
ACTIVE_LANE_CLASSES = list(LANE_CLASSES_8)
ACTIVE_LANE_CLASS_TO_ID = {name: i for i, name in enumerate(ACTIVE_LANE_CLASSES)}
ACTIVE_LANE_CLASS_SEMANTIC_RGB = list(LANE_CLASS_SEMANTIC_RGB_8)

# Default interpolation density when lines have few points (used by parseAnnotations).
LINE_INTERP_THRESHOLD = 5


def round_line_floats(line, ndigits = 6):
    line = list(line)
    for i in range(len(line)):
        line[i] = [
            round(line[i][0], ndigits),
            round(line[i][1], ndigits)
        ]
    line = tuple(line)
    return line


def configure_lane_taxonomy(class_mode: int):
    if class_mode == 8:
        classes = list(LANE_CLASSES_8)
        colors = list(LANE_CLASS_SEMANTIC_RGB_8)
    elif class_mode == 3:
        classes = list(LANE_CLASSES_3)
        colors = list(LANE_CLASS_SEMANTIC_RGB_3)
    else:
        raise ValueError(f"Unsupported class_mode: {class_mode}")
    class_to_id = {name: i for i, name in enumerate(classes)}
    return classes, class_to_id, colors


# Custom warning format cuz the default one is wayyyyyy too verbose
def custom_warning_format(message, category, filename, lineno, line = None):
    return f"WARNING : {message}\n"

warnings.formatwarning = custom_warning_format


# ============================== Helper functions ============================== #


def normalizeCoords(line, width, height):
    """
    Normalize the coords of line points.
    """
    return [(x / width, y / height) for x, y in line]


def interpLine(line: list, points_quota: int):
    """
    Interpolates a line of (x, y) points to have at least `point_quota` points.
    This helps with CurveLanes since most of its lines have so few points, 2~3.
    """
    if len(line) >= points_quota:
        return line

    # Extract x, y separately then parse to interp
    x = np.array([pt[0] for pt in line])
    y = np.array([pt[1] for pt in line])
    interp_x = np.interp
    interp_y = np.interp

    # Here I try to interp more points along the line, based on
    # distance between each subsequent original points. 

    # 1) Use distance along line as param (t)
    # This is Euclidian distance between each point and the one before it
    distances = np.cumsum(np.sqrt(
        np.diff(x, prepend = x[0])**2 + \
        np.diff(y, prepend = y[0])**2
    ))
    # Force first t as zero
    distances[0] = 0

    # 2) Generate new t evenly spaced along original line
    evenly_t = np.linspace(distances[0], distances[-1], points_quota)

    # 3) Interp x, y coordinates based on evenly t
    x_new = interp_x(evenly_t, distances, x)
    y_new = interp_y(evenly_t, distances, y)

    return list(zip(x_new, y_new))


def getLineAnchor(
        line, 
        new_img_height,
        verbose: bool = False
):
    """
    Determine "anchor" point of a line.
    """
    (x2, y2) = line[0]
    (x1, y1) = line[1]

    for i in range(1, len(line) - 1, 1):
        if (line[i][0] != x2) & (line[i][1] != y2):
            (x1, y1) = line[i]
            break

    if (x1 == x2) or (y1 == y2):
        if (x1 == x2):
            error_lane = "Vertical"
        elif (y1 == y2):
            error_lane = "Horizontal"
        if (verbose):
            warnings.warn(f"{error_lane} line detected: {line}, with these 2 anchors: ({x1}, {y1}), ({x2}, {y2}).")
        return (x1, None, None)
    
    a = (y2 - y1) / (x2 - x1)
    b = y1 - a * x1
    x0 = (new_img_height - b) / a

    return (x0, a, b)


def getEgoIndexes(
        anchors, 
        new_img_width,
        verbose: bool = False
):
    """
    Identifies 2 ego lanes - left and right - from a sorted list of line anchors.
    """
    for i in range(len(anchors)):
        if (anchors[i][0] >= new_img_width / 2):
            if (i == 0):
                if (verbose):
                    print("NO LINES on the LEFT side of frame. Registering FIRST 2 lines on the right side as egolines.")
                return (i, i + 1)
            
            return (i - 1, i)

    if (verbose):
        print("NO LINES on the RIGHT side of frame. Registering LAST 2 lines on the left side as egolines.")
    return (-2, -1)


def getDrivablePath(
        left_ego, right_ego, 
        new_img_height,
        y_coords_interp = False
):
    """
    Computes drivable path as midpoint between 2 ego lanes, basically the main point of this task.
    """
    drivable_path = []

    # When it's CurveLanes and we need interpolation among non-uniform y-coords
    if (y_coords_interp):
        left_ego = np.array(left_ego)
        right_ego = np.array(right_ego)
        y_coords_ASSEMBLE = np.unique(
            np.concatenate((
                left_ego[:, 1],
                right_ego[:, 1]
            ))
        )[::-1]
        left_x_interp = np.interp(
            y_coords_ASSEMBLE, 
            left_ego[:, 1][::-1], 
            left_ego[:, 0][::-1]
        )
        right_x_interp = np.interp(
            y_coords_ASSEMBLE, 
            right_ego[:, 1][::-1], 
            right_ego[:, 0][::-1]
        )
        mid_x = (left_x_interp + right_x_interp) / 2
        # Filter out those points that are not in the common vertical zone between 2 egos
        drivable_path = [
            [x, y] for x, y in list(zip(mid_x, y_coords_ASSEMBLE))
            if y <= min(left_ego[0][1], right_ego[0][1])
        ]
    else:
        # Get the normal drivable path from the longest common y-coords
        while (i <= len(left_ego) - 1 and j <= len(right_ego) - 1):
            if (left_ego[i][1] == right_ego[j][1]):
                drivable_path.append((
                    (left_ego[i][0] + right_ego[j][0]) / 2,     # Midpoint along x axis
                    left_ego[i][1]
                ))
                i += 1
                j += 1
            elif (left_ego[i][1] > right_ego[j][1]):
                i += 1
            else:
                j += 1

    # Extend drivable path to bottom edge of the frame
    if ((len(drivable_path) >= 2) and (drivable_path[0][1] < new_img_height - 1)):
        x1, y1 = drivable_path[1]
        x2, y2 = drivable_path[0]
        if (x2 == x1):
            x_bottom = x2
        else:
            a = (y2 - y1) / (x2 - x1)
            x_bottom = x2 + (new_img_height - 1 - y2) / a
        drivable_path.insert(0, (x_bottom, new_img_height - 1))

    # Extend drivable path to be on par with longest ego line
    # By making it parallel with longer ego line
    y_top = min(left_ego[-1][1], right_ego[-1][1])
    if ((len(drivable_path) >= 2) and (drivable_path[-1][1] > y_top)):
        sign_left_ego = left_ego[-1][0] - left_ego[-2][0]
        sign_right_ego = right_ego[-1][0] - right_ego[-2][0]
        sign_val = sign_left_ego * sign_right_ego
        # 2 egos going the same direction
        if (sign_val > 0):
            longer_ego = left_ego if left_ego[-1][1] < right_ego[-1][1] else right_ego
            if len(longer_ego) >= 2 and len(drivable_path) >= 2:
                x1, y1 = longer_ego[-1]
                x2, y2 = longer_ego[-2]
                if (x2 == x1):
                    x_top = drivable_path[-1][0]
                else:
                    a = (y2 - y1) / (x2 - x1)
                    x_top = drivable_path[-1][0] + (y_top - drivable_path[-1][1]) / a

                drivable_path.append((x_top, y_top))
        # 2 egos going opposite directions
        else:
            if len(drivable_path) >= 2:
                x1, y1 = drivable_path[-1]
                x2, y2 = drivable_path[-2]
                if (x2 == x1):
                    x_top = x1
                else:
                    a = (y2 - y1) / (x2 - x1)
                    x_top = x1 + (y_top - y1) / a

                drivable_path.append((x_top, y_top))

    return drivable_path


def calcLaneSegMask(
    lanes, 
    width, height,
    normalized: bool = True
):
    """
    Calculates binary segmentation mask for some lane lines.

    """

    # Create blank mask as new Image
    bin_seg = np.zeros(
        (height, width), 
        dtype = np.uint8
    )
    bin_seg_img = Image.fromarray(bin_seg)

    # Draw lines on mask
    draw = ImageDraw.Draw(bin_seg_img)
    for lane in lanes:
        if (normalized):
            lane = [
                (
                    x * width, 
                    y * height
                ) 
                for x, y in lane
            ]
        draw.line(
            lane, 
            fill = 255, 
            width = 4
        )
    
    # Convert back to numpy array
    bin_seg = np.array(
        bin_seg_img, 
        dtype = np.uint8
    )
    
    return bin_seg


def _resolve_line_label(i, payload):
    """Line i class name from LineLabels or LineClassIds (+ optional LaneClassMap)."""
    line_labels = payload.get("LineLabels") or []
    line_class_ids = payload.get("LineClassIds") or []
    lane_class_map = payload.get("LaneClassMap") or {}

    # Prefer explicit class ids from converter output, then fallback to labels.
    # Some datasets keep generic/free-form LineLabels which can collapse colors.
    if i < len(line_class_ids):
        cid = line_class_ids[i]
        if isinstance(cid, int) and 0 <= cid < len(ACTIVE_LANE_CLASSES):
            return ACTIVE_LANE_CLASSES[cid]
        if isinstance(cid, str) and cid.strip() in ACTIVE_LANE_CLASS_TO_ID:
            return cid.strip()
        if isinstance(cid, str):
            try:
                cid_int = int(cid.strip())
                if 0 <= cid_int < len(ACTIVE_LANE_CLASSES):
                    return ACTIVE_LANE_CLASSES[cid_int]
            except (ValueError, TypeError):
                pass
    if isinstance(lane_class_map, dict) and lane_class_map and i < len(line_class_ids):
        cid = line_class_ids[i]
        for name, idx in lane_class_map.items():
            try:
                if int(idx) == int(cid):
                    return str(name).strip()
            except (ValueError, TypeError):
                continue
    if i < len(line_labels):
        lab = str(line_labels[i]).strip()
        if lab in ACTIVE_LANE_CLASS_TO_ID:
            return lab
    return ACTIVE_LANE_CLASSES[-1]


def semantic_id_map_to_rgb(id_map: np.ndarray) -> np.ndarray:
    """H,W uint8 with values 0..N -> H,W,3 RGB for visualization."""
    h, w = id_map.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cid in range(1, len(ACTIVE_LANE_CLASS_SEMANTIC_RGB) + 1):
        rgb[id_map == cid] = ACTIVE_LANE_CLASS_SEMANTIC_RGB[cid - 1]
    return rgb


def build_semantic_lane_mask(lines, labels, width, height):
    """
    H,W uint8: 0 = background, 1..N = lane class index + 1 (active class taxonomy).
    Later-drawn lanes overwrite earlier pixels on overlap.
    """
    if len(lines) != len(labels):
        raise ValueError("lines and labels length mismatch")
    out = np.zeros((height, width), dtype=np.uint8)
    order = sorted(
        range(len(lines)),
        key=lambda i: (ACTIVE_LANE_CLASS_TO_ID.get(labels[i], len(ACTIVE_LANE_CLASSES) - 1), i),
    )
    for i in order:
        line = lines[i]
        lab = labels[i]
        cid = ACTIVE_LANE_CLASS_TO_ID.get(lab, len(ACTIVE_LANE_CLASSES) - 1) + 1
        binm = calcLaneSegMask([line], width, height, normalized=False)
        out = np.where(binm > 0, np.uint8(cid), out)
    return out


def _compute_lane_view_for_visualization(lanes_by_class, width, height):
    """
    Build legacy visualization fields from active class lanes:
      - lanes: normalized polylines
      - ego_indexes: indices in `lanes`
      - drivable_path: normalized polyline
    """
    lanes = []
    # Keep stop lines out of ego/drivable estimation.
    for cname in ACTIVE_LANE_CLASSES:
        if cname == "stop_line":
            continue
        lanes.extend(lanes_by_class.get(cname, []))

    if len(lanes) < 2:
        return lanes, [], []

    # Anchor each lane at image bottom and sort from left->right.
    anchored = []
    for idx, line in enumerate(lanes):
        denorm = [(x * width, y * height) for x, y in line]
        if len(denorm) < 2:
            continue
        x0, _, _ = getLineAnchor(denorm, height, verbose=False)
        if x0 is None or (not np.isfinite(x0)):
            continue
        anchored.append((idx, float(x0)))

    if len(anchored) < 2:
        return lanes, [], []

    anchored.sort(key=lambda t: t[1])
    x_only = [(x0, None, None) for _, x0 in anchored]
    left_i, right_i = getEgoIndexes(x_only, width, verbose=False)
    if left_i < 0:
        left_i += len(anchored)
    if right_i < 0:
        right_i += len(anchored)
    if left_i < 0 or right_i < 0 or left_i >= len(anchored) or right_i >= len(anchored):
        return lanes, [], []
    if left_i == right_i:
        return lanes, [], []

    ego_left_idx = anchored[left_i][0]
    ego_right_idx = anchored[right_i][0]
    ego_indexes = [ego_left_idx, ego_right_idx]

    left_lane_px = [(x * width, y * height) for x, y in lanes[ego_left_idx]]
    right_lane_px = [(x * width, y * height) for x, y in lanes[ego_right_idx]]
    drivable_path_px = getDrivablePath(left_lane_px, right_lane_px, height, y_coords_interp=True)

    drivable_path = []
    for x, y in drivable_path_px:
        if width > 0 and height > 0:
            drivable_path.append((x / width, y / height))

    return lanes, ego_indexes, drivable_path


def annotateGT(
        raw_img, anno_entry,
        raw_dir, mask_dir, visualization_dir,
        init_img_width, init_img_height,
        normalized = True,
        resize = None,
        crop = None,
):
    """
    Annotates and saves an image with:
        - Raw image, in "output_dir/image".
        - Lane seg mask, in "output_dir/mask".
        - Annotated image with all lanes, in "output_dir/visualization".
    """

    # Define save name
    save_name = str(img_id_counter).zfill(6)

    # Load img
    raw_img = raw_img
    new_img_height = init_img_height
    new_img_width = init_img_width
    
    # Handle image resizing
    if (resize):
        new_img_height = int(new_img_height * resize)
        new_img_width = int(new_img_width * resize)
        raw_img = raw_img.resize((
            new_img_width, 
            new_img_height
        ))

    # Handle image cropping
    if (crop):
        CROP_TOP = crop["TOP"]
        CROP_RIGHT = crop["RIGHT"]
        CROP_BOTTOM = crop["BOTTOM"]
        CROP_LEFT = crop["LEFT"]
        raw_img = raw_img.crop((
            CROP_LEFT, 
            CROP_TOP, 
            new_img_width - CROP_RIGHT, 
            new_img_height - CROP_BOTTOM
        ))
        new_img_height -= (CROP_TOP + CROP_BOTTOM)
        new_img_width -= (CROP_LEFT + CROP_RIGHT)


    assert new_img_width == 1024, f"Unexpected width: {new_img_width}"
    assert new_img_height == 640, f"Unexpected height: {new_img_height}"



    # Save raw image as JPG for lighter weight
    raw_img.save(os.path.join(raw_dir, save_name + ".jpg"))

    # Draw all lanes & lines
    draw = ImageDraw.Draw(raw_img)
    lane_colors = {
        "outer_red": (255, 0, 0), 
        "ego_green": (0, 255, 0), 
        "drive_path_yellow": (255, 255, 0)
    }
    lane_w = 5
    # Draw lanes
    for idx, line in enumerate(anno_entry.get("lanes", [])):
        if (normalized):
            line = [
                (x * new_img_width, y * new_img_height) 
                for x, y in line
            ]
        if (idx in anno_entry.get("ego_indexes", [])):
            # Ego lanes, in green
            draw.line(line, fill = lane_colors["ego_green"], width = lane_w)
        else:
            # Outer lanes, in red
            draw.line(line, fill = lane_colors["outer_red"], width = lane_w)
    # Drivable path, in yellow
    drivable_path = anno_entry.get("drivable_path", [])
    if len(drivable_path) >= 2:
        if (normalized):
            drivable_renormed = [
                (x * new_img_width, y * new_img_height) 
                for x, y in drivable_path
            ]
        else:
            drivable_renormed = drivable_path
        draw.line(drivable_renormed, fill = lane_colors["drive_path_yellow"], width = lane_w)

    # Fetch seg mask: HxW class ids (0..8) or legacy HxWx3
    mask_array = np.array(anno_entry["mask"], dtype=np.uint8)
    if mask_array.ndim == 2:
        mask_rgb = semantic_id_map_to_rgb(mask_array)
        mask_img = Image.fromarray(mask_rgb)
    else:
        mask_img = Image.fromarray(mask_array).convert("RGB")

    # Save mask (PNG, lossless)
    mask_img.save(os.path.join(mask_dir, save_name + ".png"))

    # Overlay mask on raw image, ratio 1:1
    overlayed_img = Image.blend(
        raw_img, 
        mask_img, 
        alpha = 0.5
    )

    # Save visualization img (JPG)
    overlayed_img.save(os.path.join(visualization_dir, save_name + ".jpg"))

def parseAnnotations(
        anno_path, 
        init_img_width,
        init_img_height,
        crop = None,
        resize = None,
        verbose: bool = False
    ):
    """
    Parses .lines.json annotations into per-class polylines and a semantic mask.

    Expects JSON with ``Lines`` and optionally ``LineLabels`` / ``LineClassIds`` /
    ``LaneClassMap`` (as produced by convert_labelme_to_curvelanes). Without
    class fields, lanes are assigned to ``invisible_line``.

    Returns:
        ``lanes_by_class``: dict mapping each active class name to a list of
        normalized polylines (may be empty).
        ``mask``: uint8 array (H, W) with values 0..N (background + active classes).
    """
    with open(anno_path, "r") as f:
        payload = json.load(f)
    read_data = payload.get("Lines", [])
    if len(read_data) < 1:
        if verbose:
            warnings.warn(
                f"Parsing {anno_path}: no lines in raw data. Skipping this frame."
            )
        return None

    lines = []
    labels = []
    for i, line in enumerate(read_data):
        lab = _resolve_line_label(i, payload)
        if lab not in ACTIVE_LANE_CLASS_TO_ID:
            lab = ACTIVE_LANE_CLASSES[-1]
        sorted_line = sorted(
            [(float(point["x"]), float(point["y"])) for point in line],
            key=lambda x: x[1],
            reverse=True,
        )
        lines.append(sorted_line)
        labels.append(lab)

    for i in range(len(lines)):
        if len(lines[i]) < LINE_INTERP_THRESHOLD:
            lines[i] = interpLine(lines[i], LINE_INTERP_THRESHOLD)

    new_img_height = init_img_height
    new_img_width = init_img_width

    if resize:
        new_img_height = int(new_img_height * resize)
        new_img_width = int(new_img_width * resize)
        lines = [
            [(x * resize, y * resize) for (x, y) in line]
            for line in lines
        ]

    if crop:
        CROP_TOP = crop["TOP"]
        CROP_RIGHT = crop["RIGHT"]
        CROP_BOTTOM = crop["BOTTOM"]
        CROP_LEFT = crop["LEFT"]
        cropped_lines = []
        cropped_labels = []
        for line, lab in zip(lines, labels):
            clipped = [
                (x - CROP_LEFT, y - CROP_TOP)
                for x, y in line
                if (
                    (CROP_LEFT <= x <= (new_img_width - CROP_RIGHT))
                    and (CROP_TOP <= y <= (new_img_height - CROP_BOTTOM))
                )
            ]
            if len(clipped) >= 2:
                cropped_lines.append(clipped)
                cropped_labels.append(lab)
        lines = cropped_lines
        labels = cropped_labels
        new_img_height -= CROP_TOP + CROP_BOTTOM
        new_img_width -= CROP_LEFT + CROP_RIGHT

    if len(lines) < 1:
        if verbose:
            warnings.warn(
                f"Parsing {anno_path}: no lines left after cropping. Skipping this frame."
            )
        return None

    lanes_by_class = {c: [] for c in ACTIVE_LANE_CLASSES}
    for line, lab in zip(lines, labels):
        if lab not in ACTIVE_LANE_CLASS_TO_ID:
            lab = ACTIVE_LANE_CLASSES[-1]
        lanes_by_class[lab].append(line)

    mask = build_semantic_lane_mask(lines, labels, new_img_width, new_img_height)

    anno_data = {
        "lanes_by_class": {
            c: [
                normalizeCoords(line, new_img_width, new_img_height)
                for line in lanes_by_class[c]
            ]
            for c in ACTIVE_LANE_CLASSES
        },
        "mask": mask,
    }

    lanes, ego_indexes, drivable_path = _compute_lane_view_for_visualization(
        anno_data["lanes_by_class"], new_img_width, new_img_height
    )
    anno_data["lanes"] = lanes
    anno_data["ego_indexes"] = ego_indexes
    anno_data["drivable_path"] = drivable_path

    return anno_data


if __name__ == "__main__":

    # ============================== Dataset structure ============================== #

    ROOT_DIR = ""
    LIST_SPLITS = ["train", "valid"]
    IMG_DIR = "images"
    LABEL_DIR = "labels"

    # I got this result from `./EDA_imgsizes.ipynb`
    SIZE_DICT = {
        "beeg" : (2560, 1440),
        "half_beeg" : (1280, 720),
        "weird" : (1570, 660),
        "normal" : (1920, 1080),
    }

    # ========================= Target resolution =========================
    TARGET_W = 1024
    TARGET_H = 640
    #chosen to have a multiple of 32, so that we dont have to do resize or cropping for models using various output stride (like 16 or 32 (see Unet family))

    # ========================= Cropping presets =========================

    # After resize(0.5): 2560x1440 -> 1280x720
    # Also applies to native 1280x720 images
    # 1280x720 -> 1024x640
    #   width : remove 256  -> 128 left / 128 right
    #   height: remove 80   -> 56 top / 24 bottom  (top-biased)
    CROP_BEEG = {
        "TOP": 56,
        "RIGHT": 128,
        "BOTTOM": 24,
        "LEFT": 128,
    }

    # Native 1570x660 -> 1024x640
    #   width : remove 546 -> 273 left / 273 right
    #   height: remove 20  -> 10 top / 10 bottom
    CROP_WEIRD = {
        "TOP": 10,
        "RIGHT": 273,
        "BOTTOM": 10,
        "LEFT": 273,
    }

    # After resize(0.75): 1920x1080 -> 1440x810
    # 1440x810 -> 1024x640
    #   width : remove 416 -> 208 left / 208 right
    #   height: remove 170  -> 85 top / 85 bottom
    CROP_NORMAL = {
        "TOP": 85,
        "RIGHT": 208,
        "BOTTOM": 85,
        "LEFT": 208,
    }


    # ============================== Parsing args ============================== #

    parser = argparse.ArgumentParser(
        description = "Process CurveLanes dataset - PathDet groundtruth generation"
    )
    parser.add_argument(
        "--dataset_dir", 
        "-d",
        type = str, 
        help = "CurveLanes directory (should contain exactly `Curvelanes` if you get it from Kaggle)",
        required = True
    )
    parser.add_argument(
        "--output_dir", 
        "-o",
        type = str,
        help = "Output directory",
        required = True
    )
    parser.add_argument(
        "--sampling_step",
        "-s",
        type = int,
        help = "Sampling step for each split/class",
        required = False,
    )
    # For debugging only
    parser.add_argument(
        "--early_stopping",
        "-e",
        type = int,
        help = "Num. files each split/class you wanna limit, instead of whole set.",
        required = False
    )
    parser.add_argument(
        "-c",
        "--class-mode",
        type=int,
        choices=[3, 8],
        default=8,
        help="Class taxonomy mode: 8 for lane types, 3 for egoleft_lane/egoright_lane/other",
    )
    args = parser.parse_args()

    CLASS_MODE = args.class_mode
    ACTIVE_LANE_CLASSES, ACTIVE_LANE_CLASS_TO_ID, ACTIVE_LANE_CLASS_SEMANTIC_RGB = configure_lane_taxonomy(CLASS_MODE)
    print(f"Class mode set to {CLASS_MODE} with classes: {ACTIVE_LANE_CLASSES}")

    # Parse dirs
    dataset_dir = args.dataset_dir
    output_dir = args.output_dir

    # Parse sampling step
    if (args.sampling_step):
        sampling_step = args.sampling_step
    else:
        sampling_step = 1
    print(f"Sampling step set to {sampling_step}.")

    # Parse early stopping
    if (args.early_stopping):
        print(f"Early stopping set, each split/class stops after {args.early_stopping} files.")
        early_stopping = args.early_stopping
    else:
        early_stopping = None

    # Generate output structure
    """
    --output_dir
        |----image
        |----mask
        |----visualization
        |----drivable_path.json
    """
    list_subdirs = [
        "image", 
        "mask",
        "visualization"
    ]
    if (os.path.exists(output_dir)):
        warnings.warn(f"Output directory {output_dir} already exists. Purged")
        shutil.rmtree(output_dir)
    for subdir in list_subdirs:
        subdir_path = os.path.join(output_dir, subdir)
        if (not os.path.exists(subdir_path)):
            os.makedirs(subdir_path, exist_ok = True)

    # ============================== Parsing annotations ============================== #

    # Parse data by batch
    data_master = {}
    img_id_counter = -1

    for split in LIST_SPLITS:
        print(f"\n==================== Processing {split} data ====================\n")
        raw_img_book = os.path.join(dataset_dir, ROOT_DIR, split, f"{split}.txt")
        with open(raw_img_book, "r") as f:
            list_raw_files = f.readlines()

            for i in tqdm(
                range(0, len(list_raw_files), sampling_step),
                desc = "Processing images: ",
                colour = "green"
            ):
                img_path = os.path.join(dataset_dir, ROOT_DIR, split, list_raw_files[i]).strip()
                img_id_counter += 1

                # Preload image file for multiple uses later
                raw_img = Image.open(img_path).convert("RGB")
                img_width, img_height = raw_img.size

                init_img_size = raw_img.size

                resize = None
                crop = None

                if (init_img_size == SIZE_DICT["beeg"]):
                    resize = 0.5
                    crop = CROP_BEEG
                elif (init_img_size == SIZE_DICT["half_beeg"]):
                    resize = None
                    crop = CROP_BEEG
                elif (init_img_size == SIZE_DICT["weird"]):
                    resize = None
                    crop = CROP_WEIRD
                elif (init_img_size == SIZE_DICT["normal"]):
                    resize = 0.75
                    crop = CROP_NORMAL

                anno_path = img_path.replace(".jpg", ".lines.json").replace(IMG_DIR, LABEL_DIR)

                this_data = parseAnnotations(
                    anno_path = anno_path,
                    init_img_width = img_width,
                    init_img_height = img_height,
                    resize = resize,
                    crop = crop
                )
                if (this_data is not None):

                    annotateGT(
                        raw_img = raw_img,
                        anno_entry = this_data,
                        raw_dir = os.path.join(output_dir, "image"),
                        mask_dir = os.path.join(output_dir, "mask"),
                        visualization_dir = os.path.join(output_dir, "visualization"),
                        init_img_height = img_height,
                        init_img_width = img_width,
                        resize = resize,
                        crop = crop
                    )

                    # Save as 6-digit incremental index
                    img_index = str(str(img_id_counter).zfill(6))
                    data_master[img_index] = {}
                    for c in ACTIVE_LANE_CLASSES:
                        data_master[img_index][c] = [
                            round_line_floats(line) for line in this_data["lanes_by_class"][c]
                        ]
                    data_master[img_index]["lanes"] = [
                        round_line_floats(line) for line in this_data.get("lanes", [])
                    ]
                    data_master[img_index]["ego_indexes"] = list(this_data.get("ego_indexes", []))
                    data_master[img_index]["drivable_path"] = [
                        round_line_floats([pt])[0] for pt in this_data.get("drivable_path", [])
                    ]

                    # Early stopping, it defined
                    if (early_stopping and img_id_counter >= early_stopping - 1):
                        break

    # Save master data
    with open(os.path.join(output_dir, "drivable_path.json"), "w") as f:
        json.dump(data_master, f, indent = 4)