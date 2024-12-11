from collections import deque
import os
import time

import numpy as np
import pandas as pd
import cv2


def ensure_path_exists(path):
    """
    Simple util func, make path if not exists.
        Returns path
    """
    if not os.path.exists(path):
        os.makedirs(path, 0o755, True)
    return path


def save_results_to_csv(out_path, results, mode):
    if mode == "detection":
        columns = ["frame", "x1", "y1", "x2", "y2", "class_name", "confidence"]
    else:
        columns = [
            "frame",
            "x1",
            "y1",
            "x2",
            "y2",
            "class_name",
            "confidence",
            "track_id",
        ]

    df = pd.DataFrame(results, columns=columns)
    df.to_csv(out_path, index=False)


def remove_temp_video(video_path: str):
    if os.path.exists(video_path):
        os.remove(video_path)
        print(f"Removed old video at path: {video_path}")


def hex_to_bgr(hex_color):
    # just hex digits
    hex_str = hex_color.lstrip("#")
    # slices like rr|gg|bb, then converts hex into bgr (opencv uses bgr ordering), returns tuple: (r, g, b)
    return tuple(int(hex_str[i : i + 2], 16) for i in (4, 2, 0))


def calc_progress_metrics(frame_idx, total_frames, start_time):
    elapsed_time = time.time() - start_time

    fps = frame_idx / elapsed_time if elapsed_time > 0 else 0
    remaining_frames = total_frames - frame_idx
    eta_seconds = remaining_frames / fps if fps > 0 else 0
    eta_minutes = int(eta_seconds // 60)
    eta_seconds = int(eta_seconds % 60)

    progress = frame_idx / total_frames if total_frames > 0 else 0

    return (
        progress,
        f"Running inference... {int(fps)} Frames Per Second, {int(progress*100)}% Completed, {str(eta_minutes).zfill(2)}:{str(eta_seconds).zfill(2)} Until Complete",
    )


def process_frame_detections(frame_num, detections, model_names, is_tracking=False):
    """
    Process a single frame's detections and ensure only one object is marked as 'queen'.

    Parameters:
        frame_number (int): The frame number (1-based).
        detections (list): A list of detections.
        model_names (list): A list of class names indexed by class ID.

    Returns:
        list of tuples in the format:
                        detection: (frame_number, x1, y1, x2, y2, label, conf)
                        tracking:  (frame_number, x1, y1, x2, y2, label, conf, track_id)
    """
    boxes = detections.boxes
    frame_nums = np.full(len(boxes), frame_num)
    cls_tensor = boxes.cls.cpu().numpy()
    conf_tensor = boxes.conf
    xyxy_tensor = boxes.xyxy

    # convert class ids to names
    class_names = np.array([model_names[int(i)] for i in cls_tensor])

    # find queen idxs
    queen_mask = class_names == "queen"
    if queen_mask.any():
        queen_confs = conf_tensor[queen_mask]
        best_queen_idx = queen_mask.nonzero()[0][queen_confs.argmax()]

        # create results array
        if is_tracking:
            track_ids = boxes.id.cpu().numpy()
            results = np.column_stack(
                (
                    frame_nums,
                    xyxy_tensor.cpu().numpy(),
                    np.where(np.arange(len(boxes)) == best_queen_idx, "queen", "bee"),
                    conf_tensor.cpu().numpy(),
                    track_ids,
                )
            )
        else:
            results = np.column_stack(
                (
                    frame_nums,
                    xyxy_tensor.cpu().numpy(),
                    np.where(np.arange(len(boxes)) == best_queen_idx, "queen", "bee"),
                    conf_tensor.cpu().numpy(),
                )
            )
    else:
        # all bees case
        if is_tracking:
            track_ids = boxes.id.cpu().numpy()
            results = np.column_stack(
                (
                    frame_nums,
                    xyxy_tensor.cpu().numpy(),
                    np.full(len(boxes), "bee"),
                    conf_tensor.cpu().numpy(),
                    track_ids,
                )
            )
        else:
            results = np.column_stack(
                (
                    frame_nums,
                    xyxy_tensor.cpu().numpy(),
                    np.full(len(boxes), "bee"),
                    conf_tensor.cpu().numpy(),
                )
            )

    return results


def render_boxes(frame, frame_results, bee_color, queen_color):
    boxes = frame_results[:, 1:5]
    classes = frame_results[:, 5]

    colors = [bee_color if cls == "bee" else queen_color for cls in classes]

    for box, color in zip(boxes, colors):
        x1, y1, x2, y2 = np.array(box, dtype=np.float32).astype(np.int32)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    return frame


def update_track_paths(frame_idx, frame_detections, track_paths):
    boxes = frame_detections[:, 1:5].astype(float)
    centers = np.column_stack(
        [
            (boxes[:, 0] + boxes[:, 2]) / 2,
            (boxes[:, 1] + boxes[:, 3]) / 2,
        ]
    )

    track_ids = frame_detections[:, 7].astype(float).astype(int)

    for tid, center in zip(track_ids, centers):
        if tid not in track_paths:
            track_paths[tid] = {
                "points": deque(maxlen=50),
                "last_seen_frame": frame_idx,
            }
        track_paths[tid]["points"].append(center)
        track_paths[tid]["last_seen_frame"] = frame_idx

    # clean up expired tids
    expired_ids = [
        tid
        for tid, data in track_paths.items()
        if (frame_idx - data["last_seen_frame"]) > 30
    ]
    for tid in expired_ids:
        del track_paths[tid]

    return track_paths


def draw_paths_on_frame(frame, track_paths, path_color):
    if not track_paths:
        return frame

    for data in track_paths.values():
        points = np.array(data["points"])
        if len(points) > 1:
            points = points.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [points], False, path_color, 3)
    return frame
