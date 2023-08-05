from typing import List
import numpy as np
from tqdm.notebook import tqdm
from ultralytics import YOLO
from onemetric.cv.utils.iou import box_iou_batch
from dataclasses import dataclass
from supervision.draw.color import ColorPalette
from supervision.geometry.dataclasses import Point
from supervision.video.dataclasses import VideoInfo
from supervision.video.source import get_video_frames_generator
from supervision.video.sink import VideoSink
from supervision.notebook.utils import show_frame_in_notebook
from supervision.tools.detections import Detections, BoxAnnotator
import pdb

import os
HOME = os.getcwd()
print(HOME)

import sys
sys.path.append(f"{HOME}/ByteTrack")
from yolox.tracker.byte_tracker import BYTETracker, STrack

# SOURCE_VIDEO_PATH = f"{HOME}/video1.mp4"
# TARGET_VIDEO_PATH = f"{HOME}/video1-result.mp4"

# Check if a filename is provided as a command-line argument
if len(sys.argv) != 2:
    print("Usage: python3 tracking.py <input_video_filename>")
    sys.exit(1)

# Get the video filename from the command-line arguments
video_filename = sys.argv[1]

# Generate the source and target video paths using the provided filename
SOURCE_VIDEO_PATH = os.path.join(HOME, video_filename)
TARGET_VIDEO_PATH = os.path.join(HOME, f"result-{video_filename}")

# Now you can use SOURCE_VIDEO_PATH and TARGET_VIDEO_PATH in your script
print(f"Source video path: {SOURCE_VIDEO_PATH}")
print(f"Target video path: {TARGET_VIDEO_PATH}")


@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.9
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False


# converts Detections into format that can be consumed by match_detections_with_tracks function
def detections2boxes(detections: Detections) -> np.ndarray:
    return np.hstack((
        detections.xyxy,
        detections.confidence[:, np.newaxis]
    ))


# converts List[STrack] into format that can be consumed by match_detections_with_tracks function
def tracks2boxes(tracks: List[STrack]) -> np.ndarray:
    return np.array([
        track.tlbr
        for track
        in tracks
    ], dtype=float)


# matches our bounding boxes with predictions
def match_detections_with_tracks(
    detections: Detections,
    tracks: List[STrack]
) -> Detections:
    if not np.any(detections.xyxy) or len(tracks) == 0:
        return np.empty((0,))

    tracks_boxes = tracks2boxes(tracks=tracks)
    iou = box_iou_batch(tracks_boxes, detections.xyxy)
    track2detection = np.argmax(iou, axis=1)

    tracker_ids = [None] * len(detections)

    for tracker_index, detection_index in enumerate(track2detection):
        if iou[tracker_index, detection_index] != 0:
            tracker_ids[detection_index] = tracks[tracker_index].track_id

    return tracker_ids


model = YOLO('yolov8n-pose.pt')
model.fuse()

# dict maping class_id to class_name
CLASS_NAMES_DICT = model.model.names
# class_ids of interest - car, motorcycle, bus and truck
CLASS_ID = [0] # 0 para personas


# create BYTETracker instance
byte_tracker = BYTETracker(BYTETrackerArgs())
# create VideoInfo instance
video_info = VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
# create frame generator
generator = get_video_frames_generator(SOURCE_VIDEO_PATH)
# create instance of BoxAnnotator and LineCounterAnnotator
box_annotator = BoxAnnotator(color=ColorPalette(), thickness=2, text_thickness=1, text_scale=0.5)


follow_det = -1

# open target video file
with VideoSink(TARGET_VIDEO_PATH, video_info) as sink:
    # loop over video frames
    # counter = 0
    for frame in tqdm(generator, total=video_info.total_frames):
        # model prediction on single frame and conversion to supervision Detections
        results = model(frame,conf=0.7)
        detections = Detections(
            xyxy=results[0].boxes.xyxy.cpu().numpy(),
            confidence=results[0].boxes.conf.cpu().numpy(),
            class_id=results[0].boxes.cls.cpu().numpy().astype(int)
        )
        # filtering out detections with unwanted classes
        mask = np.array([class_id in CLASS_ID for class_id in detections.class_id], dtype=bool)
        detections.filter(mask=mask, inplace=True)
        # tracking detections
        tracks = byte_tracker.update(
            output_results=detections2boxes(detections=detections),
            img_info=frame.shape,
            img_size=frame.shape
        )
        tracker_id = match_detections_with_tracks(detections=detections, tracks=tracks)
        detections.tracker_id = np.array(tracker_id)
        # filtering out detections without trackers
        mask = np.array([tracker_id is not None for tracker_id in detections.tracker_id], dtype=bool)
        detections.filter(mask=mask, inplace=True)

        # ##let's say to only track the third person:
        # if follow_det == -1 and detections.tracker_id:
        #     follow_det = detections.tracker_id[0]
        # mask = np.array([int(tracker_id) == follow_det for tracker_id in detections.tracker_id], dtype=bool)
        # detections.filter(mask=mask, inplace=True)


        # format custom labels

        # f"#{tracker_id} {CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
        labels = [
            f"{CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, tracker_id
            in detections
        ]
        # annotate and display frame
        frame = box_annotator.annotate(frame=frame, detections=detections, labels=labels)
        sink.write_frame(frame)

        # print(detections.xyxy)

        # pdb.set_trace()
        # if counter ==10:
        #     break
        # else:
        #   counter += 1
