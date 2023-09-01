from tracking import process_video_with_detection

methods = ["nearby-hand", "touching-boxes", "nearby-hand-ov", "nearby-hand-q"]

video_path = "./static/uploads/example.mp4"
for method in methods:
	process_video_with_detection(video_path, method=method)