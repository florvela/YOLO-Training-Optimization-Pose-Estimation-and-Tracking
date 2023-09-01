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
import cv2
from pathlib import Path
from typing import Any, Dict, Tuple
import openvino.runtime as ov
from ultralytics.yolo.utils import ops
import ipywidgets as widgets
from openvino.runtime import Core, Model
from typing import Tuple, Dict
import cv2
import numpy as np
from PIL import Image
from ultralytics.yolo.utils.plotting import colors
import time


core = Core()

device = "CPU"

sys.path.append(f"{os.getcwd()}/ByteTrack")
from yolox.tracker.byte_tracker import BYTETracker, STrack

MODEL_NAME = "best"
base_dir = os.path.dirname(os.getcwd())
models_dir = os.path.join(base_dir, "002-Training-models/train_results/chosen_models")
foldername = "model_yolov8_tf_yolov8m_imgsz_640_epochs_100_batch_16_dataset_v2_loss_SGD_lr_01"
weights_dir = os.path.join(models_dir, foldername, "weights")


gun_model = YOLO(os.path.join(weights_dir,f'{MODEL_NAME}.pt'))
label_map = gun_model.model.names
gun_model.fuse()


def prepare_openvino_model(model: YOLO, model_name: str, weights_dir: str) -> Tuple[ov.Model, Path]:
    model_path = Path(f"{weights_dir}/{model_name}_openvino_model/{model_name}.xml")
    model.export(format="openvino", dynamic=True, half=False)

    model = ov.Core().read_model(model_path)
    return model, model_path


det_model_temp = YOLO(os.path.join(weights_dir, f"{MODEL_NAME}.pt"))
det_ov_model, det_model_path = prepare_openvino_model(det_model_temp, MODEL_NAME, weights_dir)

det_ov_model = core.read_model(det_model_path)
if device != "CPU":
    det_ov_model.reshape({0: [1, 3, 640, 640]})
det_compiled_model = core.compile_model(det_ov_model, device)


def plot_one_box(box:np.ndarray, img:np.ndarray, color:Tuple[int, int, int] = None, mask:np.ndarray = None, label:str = None, line_thickness:int = 5):
    """
    Helper function for drawing single bounding box on image
    Parameters:
        x (np.ndarray): bounding box coordinates in format [x1, y1, x2, y2]
        img (no.ndarray): input image
        color (Tuple[int, int, int], *optional*, None): color in BGR format for drawing box, if not specified will be selected randomly
        mask (np.ndarray, *optional*, None): instance segmentation mask polygon in format [N, 2], where N - number of points in contour, if not provided, only box will be drawn
        label (str, *optonal*, None): box label string, if not provided will not be provided as drowing result
        line_thickness (int, *optional*, 5): thickness for box drawing lines
    """
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    if mask is not None:
        image_with_mask = img.copy()
        mask
        cv2.fillPoly(image_with_mask, pts=[mask.astype(int)], color=color)
        img = cv2.addWeighted(img, 0.5, image_with_mask, 0.5, 1)
    return img


def letterbox(img: np.ndarray, new_shape:Tuple[int, int] = (640, 640), color:Tuple[int, int, int] = (114, 114, 114), auto:bool = False, scale_fill:bool = False, scaleup:bool = False, stride:int = 32):
    """
    Resize image and padding for detection. Takes image as input, 
    resizes image to fit into new shape with saving original aspect ratio and pads it to meet stride-multiple constraints
    
    Parameters:
      img (np.ndarray): image for preprocessing
      new_shape (Tuple(int, int)): image size after preprocessing in format [height, width]
      color (Tuple(int, int, int)): color for filling padded area
      auto (bool): use dynamic input size, only padding for stride constrins applied
      scale_fill (bool): scale image to fill new_shape
      scaleup (bool): allow scale image if it is lower then desired input size, can affect model accuracy
      stride (int): input padding stride
    Returns:
      img (np.ndarray): image after preprocessing
      ratio (Tuple(float, float)): hight and width scaling ratio
      padding_size (Tuple(int, int)): height and width padding size
    
    
    """
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scale_fill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def preprocess_image(img0: np.ndarray):
    """
    Preprocess image according to YOLOv8 input requirements. 
    Takes image in np.array format, resizes it to specific size using letterbox resize and changes data layout from HWC to CHW.
    
    Parameters:
      img0 (np.ndarray): image for preprocessing
    Returns:
      img (np.ndarray): image after preprocessing
    """
    # resize
    img = letterbox(img0)[0]
    
    # Convert HWC to CHW
    img = img.transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    return img


def image_to_tensor(image:np.ndarray):
    """
    Preprocess image according to YOLOv8 input requirements. 
    Takes image in np.array format, resizes it to specific size using letterbox resize and changes data layout from HWC to CHW.
    
    Parameters:
      img (np.ndarray): image for preprocessing
    Returns:
      input_tensor (np.ndarray): input tensor in NCHW format with float32 values in [0, 1] range 
    """
    input_tensor = image.astype(np.float32)  # uint8 to fp32
    input_tensor /= 255.0  # 0 - 255 to 0.0 - 1.0
    
    # add batch dimension
    if input_tensor.ndim == 3:
        input_tensor = np.expand_dims(input_tensor, 0)
    return input_tensor


try:
    scale_segments = ops.scale_segments
except AttributeError:
    scale_segments = ops.scale_coords

def postprocess(
    pred_boxes:np.ndarray, 
    input_hw:Tuple[int, int], 
    orig_img:np.ndarray, 
    min_conf_threshold:float = 0.25, 
    nms_iou_threshold:float = 0.7, 
    agnosting_nms:bool = False, 
    max_detections:int = 300,
    pred_masks:np.ndarray = None,
    retina_mask:bool = False
):
    """
    YOLOv8 model postprocessing function. Applied non maximum supression algorithm to detections and rescale boxes to original image size
    Parameters:
        pred_boxes (np.ndarray): model output prediction boxes
        input_hw (np.ndarray): preprocessed image
        orig_image (np.ndarray): image before preprocessing
        min_conf_threshold (float, *optional*, 0.25): minimal accepted confidence for object filtering
        nms_iou_threshold (float, *optional*, 0.45): minimal overlap score for removing objects duplicates in NMS
        agnostic_nms (bool, *optiona*, False): apply class agnostinc NMS approach or not
        max_detections (int, *optional*, 300):  maximum detections after NMS
        pred_masks (np.ndarray, *optional*, None): model ooutput prediction masks, if not provided only boxes will be postprocessed
        retina_mask (bool, *optional*, False): retina mask postprocessing instead of native decoding
    Returns:
       pred (List[Dict[str, np.ndarray]]): list of dictionary with det - detected boxes in format [x1, y1, x2, y2, score, label] and segment - segmentation polygons for each element in batch
    """
    nms_kwargs = {"agnostic": agnosting_nms, "max_det":max_detections}
    # if pred_masks is not None:
    #     nms_kwargs["nm"] = 32
    preds = ops.non_max_suppression(
        torch.from_numpy(pred_boxes),
        min_conf_threshold,
        nms_iou_threshold,
        nc=1,
        **nms_kwargs
    )
    results = []
    proto = torch.from_numpy(pred_masks) if pred_masks is not None else None

    for i, pred in enumerate(preds):
        shape = orig_img[i].shape if isinstance(orig_img, list) else orig_img.shape
        if not len(pred):
            results.append({"det": [], "segment": []})
            continue
        if proto is None:
            pred[:, :4] = ops.scale_boxes(input_hw, pred[:, :4], shape).round()
            results.append({"det": pred})
            continue
        if retina_mask:
            pred[:, :4] = ops.scale_boxes(input_hw, pred[:, :4], shape).round()
            masks = ops.process_mask_native(proto[i], pred[:, 6:], pred[:, :4], shape[:2])  # HWC
            segments = [scale_segments(input_hw, x, shape, normalize=False) for x in ops.masks2segments(masks)]
        else:
            masks = ops.process_mask(proto[i], pred[:, 6:], pred[:, :4], input_hw, upsample=True)
            pred[:, :4] = ops.scale_boxes(input_hw, pred[:, :4], shape).round()
            segments = [scale_segments(input_hw, x, shape, normalize=False) for x in ops.masks2segments(masks)]
        results.append({"det": pred[:, :6].numpy(), "segment": segments})
    return results


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


def write_json_file(file_path, data):
    with open(file_path, 'w') as file:
        json.dump(data, file)


def read_json_file(file_path):
    if not os.path.exists(file_path):
        return {}

    with open(file_path, 'r') as file:
        return json.load(file)


def convert_avi_to_mp4(avi_file_path, output_name):
    os.popen("ffmpeg -i '{input}' -ac 2 -b:v 2000k -c:a aac -c:v libx264 -b:a 160k -vprofile high -bf 0 -strict experimental -f mp4 '{output}.mp4'".format(input = avi_file_path, output = output_name))
    return True


def process_video_with_detection(video_path, output_directory="static/results", method="nearby-hand"):

    SOURCE_VIDEO_PATH = os.path.join(os.getcwd(), video_path)
    tracking_key = os.path.basename(video_path).replace(".mp4", "_" + method + ".mp4")

    TARGET_VIDEO_PATH = os.path.join(output_directory, tracking_key)

    print(f"Source video path: {SOURCE_VIDEO_PATH}")
    print(f"Target video path: {TARGET_VIDEO_PATH}")

    file_path = 'data.json'
    data = read_json_file(file_path)

    tracking_data = dict()

    tracking_data["optimized"] = False

    if method == "nearby-hand":
        tracking_data["method"] = "Tracked person with the closest wrist to detected guns"
    elif method == "nearby-hand-ov":
        tracking_data["method"] = "Tracked person with the closest wrist to detected guns"
        tracking_data["optimized"] = True
    else:
        tracking_data["method"] = "Track person with the bounding boxes near detected guns"
    
    def detect(image:np.ndarray, model:Model):
        """
        OpenVINO YOLOv8 model inference function. Preprocess image, runs model inference and postprocess results using NMS.
        Parameters:
            image (np.ndarray): input image.
            model (Model): OpenVINO compiled model.
        Returns:
            detections (np.ndarray): detected boxes in format [x1, y1, x2, y2, score, label]
        """
        num_outputs = len(model.outputs)
        preprocessed_image = preprocess_image(image)
        input_tensor = image_to_tensor(preprocessed_image)
        result = model(input_tensor)
        boxes = result[model.output(0)]
        masks = None
        if num_outputs > 1:
            masks = result[model.output(1)]
        input_hw = input_tensor.shape[2:]
        detections = postprocess(pred_boxes=boxes, input_hw=input_hw, orig_img=image, pred_masks=masks)
        return detections

    # model = YOLO(os.path.join(os.getcwd(),'weights/yolov8n-pose.pt'))
    model = YOLO('yolov8n-pose.pt')
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

    counter = 0
    # method = "nearby-hand-ov"


    with VideoSink(TARGET_VIDEO_PATH, video_info) as sink:
        pose_speed = list()
        det_speed = list()
        if method == "nearby-hand-ov":
            # loop over video frames
            for frame in tqdm(generator, total=video_info.total_frames):
                counter += 1
                print(f"{counter}/{video_info.total_frames}")

                input_image = np.array(frame)
                start = time.time()
                gun_results = detect(input_image, det_compiled_model)[0]
                end = time.time()
                det_speed.append((end - start)*1000)

                # model prediction on single frame and conversion to supervision Detections
                results = model(frame, conf=0.7) #, verbose=False)
                frame = results[0].plot(boxes=False)
                speed = results[0].speed
                pose_speed.append(speed['preprocess'] + speed['inference'] + speed['postprocess'])


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
                if len(gun_results["det"]) and len(results[0]):
                    # coordenadas x,y de las munecas son 10 y 9
                    result_hand_keypoints = [sub_array[[9,10]] for sub_array in results[0].keypoints.xy]

                    for bounding_box in gun_results["det"]:
                        keep_track.append(find_closest_keypoint(result_hand_keypoints, bounding_box[0:4]))

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
                if len(gun_results["det"]):
                    for res in gun_results["det"]:
                        detections2 = Detections(
                            xyxy=res[0:4].cpu().numpy().reshape(1, 4),
                            confidence=res[4].cpu().numpy().reshape(1),
                            class_id=res[5].cpu().numpy().astype(int).reshape(1)
                        )
                        labels = [
                            f"gun {confidence:0.2f}"
                            for _, confidence, class_id, _
                            in detections2
                        ]
                        frame = box_annotator.annotate(frame=frame, detections=detections2, labels=labels)

                sink.write_frame(frame)

        elif method == "nearby-hand":
            for frame in tqdm(generator, total=video_info.total_frames):
                counter += 1
                print(f"{counter}/{video_info.total_frames}")
                gun_results = gun_model(frame, conf=0.75)
                speed = gun_results[0].speed
                det_speed.append(speed['preprocess'] + speed['inference'] + speed['postprocess'])

                # model prediction on single frame and conversion to supervision Detections
                results = model(frame, conf=0.7)
                frame = results[0].plot(boxes=False)
                speed = results[0].speed
                pose_speed.append(speed['preprocess'] + speed['inference'] + speed['postprocess'])


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
                counter += 1
                print(f"{counter}/{video_info.total_frames}")
                gun_results = gun_model(frame, conf=0.5)
                speed = gun_results[0].speed
                det_speed.append(speed['preprocess'] + speed['inference'] + speed['postprocess'])

                # model prediction on single frame and conversion to supervision Detections
                results = model(frame, conf=0.7)
                frame = results[0].plot(boxes=False)
                speed = results[0].speed
                pose_speed.append(speed['preprocess'] + speed['inference'] + speed['postprocess'])

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
        tracking_data["pose_average_speed(ms)"] = sum(pose_speed) / len(pose_speed)
        tracking_data["pose_FPS"] = 1000 / tracking_data["pose_average_speed(ms)"] 
        
        tracking_data["det_average_speed(ms)"] = sum(det_speed) / len(det_speed)
        tracking_data["det_FPS"] = 1000 / tracking_data["det_average_speed(ms)"] 

        data[tracking_key] = tracking_data

        sink.writer.release()
        write_json_file(file_path, data)

