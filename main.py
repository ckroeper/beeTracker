from collections import defaultdict, deque
import os
import tempfile
import time

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

from utils import (
    calc_progress_metrics,
    render_boxes,
    draw_paths_on_frame,
    ensure_path_exists,
    hex_to_bgr,
    process_frame_detections,
    remove_temp_video,
    save_results_to_csv,
    update_track_paths,
)


@st.cache_resource
def load_model():
    print("Loading model")

    return YOLO("models/yolo11l-bee.pt")


# for better performance, play around with batch_sz
# increasing can give better performance, but uses more vram
def run_inference(
    model,
    input_path,
    mode="detection",
    progress_callback=None,
    stop_flag=lambda: False,
    batch_sz=8,
):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        st.error("Error opening video file.")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    results = []

    frame_idx = 0
    batch_frames, batch_idxs = [], []  # vars for batch mode

    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        if stop_flag():
            cap.release()
            st.toast(
                body=f"Canceling inference, returning the results up to frame {frame_idx}",
                icon="⚠️",  # warning sign
            )
            st.session_state.results = results

        # Run detect or track mode on current frame
        if mode == "detection":
            batch_frames.append(frame)
            batch_idxs.append(frame_idx)
            frame_idx += 1

            if len(batch_frames) != batch_sz:
                continue

            # run batched inference
            detection_results = model(batch_frames)

            for idx, detection_result in zip(batch_idxs, detection_results):
                frame_results = process_frame_detections(
                    frame_num=idx + 1,
                    detections=detection_result,
                    model_names=model.names,
                )
                results.append(frame_results)

            # reset batch
            batch_frames, batch_idxs = [], []

            # Update progress
            if progress_callback:
                progress, text = calc_progress_metrics(
                    frame_idx, total_frames, start_time
                )
                progress_callback(progress, text)
        else:
            # tracking mode requires frame by frame (no batching)
            track_boxes = model.track(frame, persist=True)[0]
            frame_results = process_frame_detections(
                frame_num=frame_idx + 1,
                detections=track_boxes,
                model_names=model.names,
                is_tracking=True,
            )
            results.append(frame_results)
            frame_idx += 1

            # Update progress
            if progress_callback:
                progress, text = calc_progress_metrics(
                    frame_idx, total_frames, start_time
                )
                progress_callback(progress, text)

    # if any leftover frames in batch, process them now
    if mode == "detection" and batch_frames:
        detection_results = model(batch_frames)
        for idx, detection_result in zip(batch_idxs, detection_results):
            frame_results = process_frame_detections(
                frame_num=idx + 1,
                detections=detection_result,
                model_names=model.names,
            )
            results.append(frame_results)

            if progress_callback:
                progress, text = calc_progress_metrics(
                    frame_idx, total_frames, start_time
                )
                progress_callback(progress, text)

    cap.release()
    st.toast("Video processed successfully")
    st.session_state.results = np.vstack(results)


def render_video_with_overlays(
    input_path,
    results,
    mode,
    progress_callback=None,
    draw_boxes=True,
    draw_paths=False,
    out_path="output_with_overlay.mp4",
):
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*"H264")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    bee_color = st.session_state.bee_color
    queen_color = st.session_state.queen_color
    path_color = st.session_state.queen_color

    print("Beginning to render video, saving to: ", out_path)

    # Preprocess res by frame
    frame_idxs = results[:, 0].astype(int)
    unique_frames = np.unique(frame_idxs)

    # track_paths = defaultdict(lambda: deque(maxlen=50))
    track_paths = {} if draw_paths else None

    frame_idx = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        if frame_idx in unique_frames:
            # get all dets for curr frame
            frame_mask = frame_idxs == frame_idx
            frame_dets = results[frame_mask]

            # draw boxes if enabled
            if draw_boxes:
                render_boxes(frame, frame_dets, bee_color, queen_color)

            # update and draw paths if tracking
            if mode == "tracking" and draw_paths:
                track_paths = update_track_paths(frame_idx, frame_dets, track_paths)
                frame = draw_paths_on_frame(frame, track_paths, path_color)

        out.write(frame)

        # Update progress
        if progress_callback:
            progress, text = calc_progress_metrics(frame_idx, total_frames, start_time)
            progress_callback(progress, text)

        frame_idx += 1

    cap.release()
    out.release()

    st.toast("Video created successfully")
    st.session_state.overlay_video_path = out_path


def load_video(uploaded_file):
    """
    Workaround for streamlit not being able to return paths.
    We store video in temp file and return path to temp file.
    """
    if "temp_video_path" in st.session_state and st.session_state.temp_video_path:
        remove_temp_video(st.session_state.temp_video_path)

    # Handle uploaded file if it exists
    file_extension = os.path.splitext(uploaded_file.name)[-1]
    tempVideo = tempfile.NamedTemporaryFile(suffix=file_extension, delete=False)
    tempVideo.write(uploaded_file.read())
    tempVideo.close()

    st.session_state.temp_video_path = tempVideo.name
    st.toast("Uploaded video successfully")


def main():
    st.title("BeeTrack")
    st.write(
        "Use this tool to run bee detection or tracking on your video. Upload a video, choose the mode, and run inference."
    )

    # Button to start inference
    if "last_uploaded_file" not in st.session_state:
        st.session_state.last_uploaded_file = ""
    if "inference_running" not in st.session_state:
        st.session_state.inference_running = False
    if "cancel_requested" not in st.session_state:
        st.session_state.cancel_requested = False
    if "results" not in st.session_state:
        st.session_state.results = None
    if "overlay_video" not in st.session_state:
        st.session_state.overlay_video_path = None

    # colors are in bgr
    if "path_color" not in st.session_state:
        st.session_state.path_color = (255, 255, 255)  # white
    if "bee_color" not in st.session_state:
        st.session_state.bee_color = (0, 255, 255)  # yellow
    if "queen_color" not in st.session_state:
        st.session_state.queen_color = (0, 0, 255)  # red

    # Load model once
    model = load_model()

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload or Drag a Video File Here",
        accept_multiple_files=False,
        type=[
            "mp4",
            "avi",
            "mov",
            "mkv",
        ],  # etc, you can add more supported formats from https://docs.ultralytics.com/modes/predict/#videos
        key="file_uploader",
    )

    if uploaded_file is None:
        return

    if uploaded_file.name != st.session_state.last_uploaded_file:
        load_video(uploaded_file)
        st.session_state.last_uploaded_file = uploaded_file.name

    def reset_results():
        st.session_state.results = None

    st.header("Run Model")
    mode = st.radio(
        "Mode", ["Detection", "Tracking"], on_change=reset_results, key="mode_changer"
    )

    def update_running_state():
        st.session_state.inference_running = True
        st.session_state.cancel_requested = False
        st.session_state.results = None

    st.button(
        "Begin Inference",
        disabled=st.session_state.inference_running,
        on_click=update_running_state,
    )

    # If inference is running, show the progress bar and cancel buttons
    if st.session_state.inference_running:
        progress_bar = st.progress(0, text="Running inference...")
        cancel_button = st.button("Cancel Inference")
        if cancel_button:
            st.session_state.cancel_requested = True

        # Run inference
        def update_progress(p, text):
            progress_bar.progress(p, text=text)

        def stop_flag():
            return st.session_state.cancel_requested

        run_inference(
            model,
            st.session_state.temp_video_path,
            mode=mode.lower(),
            progress_callback=update_progress,
            stop_flag=stop_flag,
        )
        st.session_state.inference_running = False

    # if inference is done and we have results, show options to save
    if st.session_state.results is not None and not st.session_state.inference_running:
        st.header("Export Results")

        def generate_name(mode):
            csv_out_dir = ensure_path_exists("./results/positions")
            input_video_name = os.path.splitext(st.session_state.last_uploaded_file)[
                0
            ]  # get only name component
            return f"{csv_out_dir}/{input_video_name}-{'detect' if mode != 'Tracking' else 'track'}-positions.csv"

        def export_on_click(mode):
            save_path = generate_name(mode)
            save_results_to_csv(save_path, st.session_state.results, mode.lower())

        bee_color = st.color_picker("Bee Color", value="#FFFF00")  # yellow
        queen_color = st.color_picker("Queen Color", value="#FF0000")  # red
        path_color_placeholder = st.empty()

        st.session_state.bee_color = hex_to_bgr(bee_color)
        st.session_state.queen_color = hex_to_bgr(queen_color)

        draw_boxes = st.checkbox("Draw Bounding Boxes", value=True)

        draw_paths = False
        if mode == "Tracking":
            draw_paths = st.checkbox("Draw Paths", value=False)
            path_color = path_color_placeholder.color_picker(
                "Path Color", value="#FFFFFF"
            )  # white
            st.session_state.path_color = hex_to_bgr(path_color)

        progress_bar_placeholder = st.empty()
        video_placeholder = st.empty()

        st.button("Export CSV", on_click=export_on_click, args=(mode,))
        st.button(
            "Export Video",
            on_click=generate_video_section,
            args=(
                mode,
                draw_boxes,
                draw_paths,
                progress_bar_placeholder,
                video_placeholder,
            ),
        )


def generate_video_section(
    mode, draw_boxes, draw_paths, progress_bar_placeholder, video_placeholder
):
    def generate_name(mode):
        vid_out_dir = ensure_path_exists("./results/videos")
        input_video_name = os.path.splitext(st.session_state.last_uploaded_file)[
            0
        ]  # get only name component
        return f"{vid_out_dir}/{input_video_name}-{'detect' if mode != 'Tracking' else 'track'}.mp4"

    # Run inference
    def update_progress(p, text):
        progress_bar_placeholder.progress(p, text=text)

    progress_bar_placeholder.progress(0, text="Creating visualization")

    save_path = generate_name(mode)
    render_video_with_overlays(
        st.session_state.temp_video_path,
        st.session_state.results,
        mode=mode.lower(),
        progress_callback=update_progress,
        draw_boxes=draw_boxes,
        draw_paths=draw_paths,
        out_path=save_path,
    )

    video_placeholder.video(
        st.session_state.overlay_video_path, autoplay=True, muted=True
    )


if __name__ == "__main__":
    st.set_page_config("BeeTrack", layout="wide")
    main()

    # # if inference is done and we have results, show options to save
    # if st.session_state.results is not None and not st.session_state.inference_running:
    #     st.header("Export Results")
    #
    #     csv_col, video_col = st.columns(2)
    #
    #     with csv_col:
    #
    #         def generate_name(mode):
    #             csv_out_dir = ensure_path_exists("./results/positions")
    #             input_video_name = os.path.splitext(
    #                 st.session_state.last_uploaded_file
    #             )[0]  # get only name component
    #             return f"{csv_out_dir}/{input_video_name}-{'detect' if mode != 'Tracking' else 'track'}-positions.csv"
    #
    #         def export_on_click(mode):
    #             save_path = generate_name(mode)
    #             save_results_to_csv(save_path, st.session_state.results, mode.lower())
    #
    #         st.button("Export CSV", on_click=export_on_click, args=(mode,))
    #
    #     with video_col:
    #         bee_color = st.color_picker("Bee Color", value="#FFFF00")  # yellow
    #         queen_color = st.color_picker("Queen Color", value="#FF0000")  # red
    #         path_color_placeholder = st.empty()
    #
    #         st.session_state.bee_color = hex_to_bgr(bee_color)
    #         st.session_state.queen_color = hex_to_bgr(queen_color)
    #
    #         draw_boxes = st.checkbox("Draw Bounding Boxes", value=True)
    #
    #         draw_paths = False
    #         if mode == "Tracking":
    #             draw_paths = st.checkbox("Draw Paths", value=False)
    #             path_color = path_color_placeholder.color_picker(
    #                 "Path Color", value="#FFFFFF"
    #             )  # white
    #             st.session_state.path_color = hex_to_bgr(path_color)
    #
    #         progress_bar_placeholder = st.empty()
    #         video_placeholder = st.empty()
    #
    #         st.button(
    #             "Create Video",
    #             on_click=generate_video_section,
    #             args=(
    #                 mode,
    #                 draw_boxes,
    #                 draw_paths,
    #                 progress_bar_placeholder,
    #                 video_placeholder,
    #             ),
    #         )
