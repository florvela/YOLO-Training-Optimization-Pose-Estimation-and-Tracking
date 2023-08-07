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
import torch
import argparse
import os
import sys

sys.path.append(f"{os.getcwd()}/ByteTrack")
from yolox.tracker.byte_tracker import BYTETracker, STrack

def parse_arguments():
    parser = argparse.ArgumentParser(description="Your script description")
    parser.add_argument("--output", default=".", help="Output directory path")
    parser.add_argument("--video", default="../videos_for_testing/example.mp4", help="Input video filename")
    parser.add_argument("--method", default="nearby-hand", help="Either nearby-hand or touching_boxes", choices=["nearby-hand", "touching-boxes"])

    args = parser.parse_args()
    return args


args = parse_arguments()

# Access the output directory path from the command-line arguments
output_directory = args.output
video_path = args.video
video_filename = os.path.basename(video_path)
method = args.method

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Generate the source and target video paths using the provided filename
SOURCE_VIDEO_PATH = os.path.join(os.getcwd(), video_path)
TARGET_VIDEO_PATH = os.path.join(output_directory, f"result-{method}-{video_filename}")
print(f"Source video path: {SOURCE_VIDEO_PATH}")
print(f"Target video path: {TARGET_VIDEO_PATH}")

gun_model = YOLO(os.path.join(os.getcwd(),'weights/guns.pt'))
gun_model.fuse()

model = YOLO(os.path.join(os.getcwd(),'weights/yolov8n-pose.pt'))
model.fuse()


@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.9
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False

# class_ids of interest - car, motorcycle, bus and truck
POSE_INTEREST_IDS = [0] # 0 para personas

# create BYTETracker instance
byte_tracker = BYTETracker(BYTETrackerArgs())
# create VideoInfo instance
video_info = VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
# create frame generator
generator = get_video_frames_generator(SOURCE_VIDEO_PATH)
# create instance of BoxAnnotator and LineCounterAnnotator
box_annotator = BoxAnnotator(color=ColorPalette(), thickness=2, text_thickness=1, text_scale=0.5)


follow_det = -1
flag = False
track_true = set()



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


# matches bounding boxes with predictions
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


def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))


def find_closest_keypoint(keypoints, bounding_box):
    # Convert bounding box to [x, y] format
    box_x = (bounding_box[0] + bounding_box[2]) / 2
    box_y = (bounding_box[1] + bounding_box[3]) / 2
    box_points = [box_x, box_y]

    min_distance = float('inf')
    closest_index = -1

    for i, keypoints_pair in enumerate(keypoints):
        for point in keypoints_pair.view(-1, 2).numpy():
            distance = euclidean_distance(point, box_points)
            if distance < min_distance:
                min_distance = distance
                closest_index = i

    return closest_index


def are_boxes_touching(box1, box2):
    x_min1, y_min1, x_max1, y_max1 = box1
    x_min2, y_min2, x_max2, y_max2 = box2

    # Check for overlap in the x-axis
    if x_max1 < x_min2 or x_min1 > x_max2:
        return False

    # Check for overlap in the y-axis
    if y_max1 < y_min2 or y_min1 > y_max2:
        return False

    # If there's an overlap in both x-axis and y-axis, the boxes are touching
    return True


# open target video file
with VideoSink(TARGET_VIDEO_PATH, video_info) as sink:
    # loop over video frames
    if method == "nearby-hand":
        for frame in tqdm(generator, total=video_info.total_frames):
            gun_results = gun_model(frame, conf=0.5)

            # model prediction on single frame and conversion to supervision Detections
            results = model(frame, conf=0.7)
            frame = results[0].plot(boxes=False)


            detections = Detections(
                xyxy=results[0].boxes.xyxy.cpu().numpy(),
                confidence=results[0].boxes.conf.cpu().numpy(),
                class_id=results[0].boxes.cls.cpu().numpy().astype(int)
            )
            
            # filtering out detections with unwanted classes
            mask = np.array([class_id in POSE_INTEREST_IDS for class_id in detections.class_id], dtype=bool)
            detections.filter(mask=mask, inplace=True)

            # tracking detections
            tracks = byte_tracker.update(
                output_results=detections2boxes(detections=detections),
                img_info=frame.shape,
                img_size=frame.shape
            )
            tracker_id = match_detections_with_tracks(detections=detections, tracks=tracks)
            detections.tracker_id = np.array(tracker_id)

            keep_track = list()
            if len(gun_results[0]) and len(results[0]):
                # coordenadas x,y de las munecas son 10 y 9
                result_hand_keypoints = [sub_array[[9,10]] for sub_array in results[0].keypoints.xy]

                for bounding_box in gun_results[0].boxes.xyxy:
                    keep_track.append(find_closest_keypoint(result_hand_keypoints, bounding_box))

            if len(detections.tracker_id):
                for tracker_id in detections.tracker_id[keep_track]:
                    track_true.add(tracker_id)

            mask = np.array([tracker_id is not None and tracker_id in track_true for tracker_id in detections.tracker_id], dtype=bool)
            detections.filter(mask=mask, inplace=True)

            labels = [
                f"#{tracker_id} {confidence:0.2f}"
                for _, confidence, class_id, tracker_id
                in detections
            ]
            # annotate and display frame
            frame = box_annotator.annotate(frame=frame, detections=detections, labels=labels)

            ### Anotar las guns!!!! 
            detections2 = Detections(
                xyxy=gun_results[0].boxes.xyxy.cpu().numpy(),
                confidence=gun_results[0].boxes.conf.cpu().numpy(),
                class_id=gun_results[0].boxes.cls.cpu().numpy().astype(int)
            )
            labels = [
                f"{gun_model.model.names[class_id]} {confidence:0.2f}"
                for _, confidence, class_id, _
                in detections2
            ]
            frame = box_annotator.annotate(frame=frame, detections=detections2, labels=labels)

            sink.write_frame(frame)
    else:
        for frame in tqdm(generator, total=video_info.total_frames):
            gun_results = gun_model(frame, conf=0.5)

            # model prediction on single frame and conversion to supervision Detections
            results = model(frame, conf=0.7)
            frame = results[0].plot(boxes=False)

            ################ Cambiar esto para ver si lo agarracon la mano y no solo si el arma esta cerca de la persona
            touching_indexes = []
            not_touching_indexes = []

            if len(gun_results[0]) > 0:
                for idx2, box2 in enumerate(results[0].boxes.xyxy):
                    for box1 in gun_results[0].boxes.xyxy:
                        if are_boxes_touching(box1, box2):
                            touching_indexes.append(idx2)
                            break  # No need to check other boxes in boxes1 for this box2
                    if idx2 not in touching_indexes:
                       not_touching_indexes.append(idx2)
            else:
                touching_indexes = []
                not_touching_indexes = []


            detections = Detections(
                xyxy=results[0].boxes.xyxy.cpu().numpy(),
                confidence=results[0].boxes.conf.cpu().numpy(),
                class_id=results[0].boxes.cls.cpu().numpy().astype(int)
            )
            
            # filtering out detections with unwanted classes
            mask = np.array([class_id in POSE_INTEREST_IDS for class_id in detections.class_id], dtype=bool)
            detections.filter(mask=mask, inplace=True)

            # tracking detections
            tracks = byte_tracker.update(
                output_results=detections2boxes(detections=detections),
                img_info=frame.shape,
                img_size=frame.shape
            )
            tracker_id = match_detections_with_tracks(detections=detections, tracks=tracks)
            detections.tracker_id = np.array(tracker_id)

            if len(detections.tracker_id):
                for tracker_id in detections.tracker_id[touching_indexes]:
                    track_true.add(tracker_id)

            mask = np.array([tracker_id is not None and tracker_id in track_true for tracker_id in detections.tracker_id], dtype=bool)
            detections.filter(mask=mask, inplace=True)

            labels = [
                f"#{tracker_id} {confidence:0.2f}"
                for _, confidence, class_id, tracker_id
                in detections
            ]
            # annotate and display frame
            frame = box_annotator.annotate(frame=frame, detections=detections, labels=labels)


            ### Anotar las guns!!!! 
            detections2 = Detections(
                xyxy=gun_results[0].boxes.xyxy.cpu().numpy(),
                confidence=gun_results[0].boxes.conf.cpu().numpy(),
                class_id=gun_results[0].boxes.cls.cpu().numpy().astype(int)
            )
            labels = [
                f"{gun_model.model.names[class_id]} {confidence:0.2f}"
                for _, confidence, class_id, _
                in detections2
            ]
            frame = box_annotator.annotate(frame=frame, detections=detections2, labels=labels)

            sink.write_frame(frame)
