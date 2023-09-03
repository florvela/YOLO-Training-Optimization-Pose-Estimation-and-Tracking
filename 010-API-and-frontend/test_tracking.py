from tracking import process_video_with_detection
import os

methods = ["nearby-hand", "touching-boxes", "nearby-hand-ov", "nearby-hand-q"]

videos_path = "../videos_for_testing/"
for video in os.listdir(videos_path):
	video_path = videos_path + video
	for method in methods:
		process_video_with_detection(video_path, method=method)