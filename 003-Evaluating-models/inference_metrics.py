from typing import List
import numpy as np
from tqdm.notebook import tqdm
from ultralytics import YOLO
from onemetric.cv.utils.iou import box_iou_batch
from dataclasses import dataclass
from supervision.draw.color import ColorPalette
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
import json


def process_video_with_detection(video_path, output_directory=".", method="nearby-hand"):

    SOURCE_VIDEO_PATH = os.path.join(os.getcwd(), video_path)
    TARGET_VIDEO_PATH = os.path.join(output_directory, os.path.basename(video_path))

    print(f"Source video path: {SOURCE_VIDEO_PATH}")
    print(f"Target video path: {TARGET_VIDEO_PATH}")

    # gun_model = YOLO('yolov8n.pt')
    gun_model = YOLO(os.path.join(os.getcwd(),'../002-Training-models/train_results/model_yolov8_tf_yolov8s_imgsz_800_epochs_78_batch_16_dataset_v2/weights/best.pt'))
    gun_model.fuse()

    # create VideoInfo instance
    video_info = VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
    # create frame generator
    generator = get_video_frames_generator(SOURCE_VIDEO_PATH)


    for frame in tqdm(generator, total=video_info.total_frames):
        gun_results = gun_model(frame, conf=0.5)
        pdb.set_trace()

        # # model prediction on single frame and conversion to supervision Detections
        # results = model(frame, conf=0.7)
        # frame = results[0].plot(boxes=False)


        # detections = Detections(
        #     xyxy=results[0].boxes.xyxy.cpu().numpy(),
        #     confidence=results[0].boxes.conf.cpu().numpy(),
        #     class_id=results[0].boxes.cls.cpu().numpy().astype(int)
        # )
        
        # # filtering out detections with unwanted classes
        # mask = np.array([class_id in POSE_INTEREST_IDS for class_id in detections.class_id], dtype=bool)
        # detections.filter(mask=mask, inplace=True)

        # # tracking detections
        # tracks = byte_tracker.update(
        #     output_results=detections2boxes(detections=detections),
        #     img_info=frame.shape,
        #     img_size=frame.shape
        # )
        # tracker_id = match_detections_with_tracks(detections=detections, tracks=tracks)
        # detections.tracker_id = np.array(tracker_id)

        # keep_track = list()
        # if len(gun_results[0]) and len(results[0]):
        #     # coordenadas x,y de las munecas son 10 y 9
        #     result_hand_keypoints = [sub_array[[9,10]] for sub_array in results[0].keypoints.xy]

        #     for bounding_box in gun_results[0].boxes.xyxy:
        #         keep_track.append(find_closest_keypoint(result_hand_keypoints, bounding_box))

        # if len(detections.tracker_id):
        #     for tracker_id in detections.tracker_id[keep_track]:
        #         track_true.add(tracker_id)

        # mask = np.array([tracker_id is not None and tracker_id in track_true for tracker_id in detections.tracker_id], dtype=bool)
        # detections.filter(mask=mask, inplace=True)

        # labels = [
        #     f"#{tracker_id} {confidence:0.2f}"
        #     for _, confidence, class_id, tracker_id
        #     in detections
        # ]
        # # annotate and display frame
        # frame = box_annotator.annotate(frame=frame, detections=detections, labels=labels)

        # ### Anotar las guns!!!! 
        # detections2 = Detections(
        #     xyxy=gun_results[0].boxes.xyxy.cpu().numpy(),
        #     confidence=gun_results[0].boxes.conf.cpu().numpy(),
        #     class_id=gun_results[0].boxes.cls.cpu().numpy().astype(int)
        # )
        # labels = [
        #     f"{gun_model.model.names[class_id]} {confidence:0.2f}"
        #     for _, confidence, class_id, _
        #     in detections2
        # ]
        # frame = box_annotator.annotate(frame=frame, detections=detections2, labels=labels)

        # sink.write_frame(frame)

process_video_with_detection("../videos_for_testing/example.mp4")