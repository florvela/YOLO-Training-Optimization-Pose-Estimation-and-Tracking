import os

cwd = os.getcwd()

for filename in os.listdir("output"):
	if filename.endswith(".mp4"):
		os.system(f"cd ../007-ByteTrack-and-YOLOv8-pose && source ./env/bin/activate && python3 tracking.py --video {cwd}/output/{filename} --output {cwd}/detection_on_removed_background/")

for detected_filename in os.listdir("./detection_on_removed_background"):
	if detected_filename.endswith("_nbs.mp4"):
		original_filename = detected_filename.replace("_nbs","").replace("result-nearby-hand-","")
		# print(original_filename)
		# print(os.path.exists("../videos_for_testing/"+original_filename))
		os.system(f"cd ../007-ByteTrack-and-YOLOv8-pose && source ./env/bin/activate && python3 tracking.py --video {cwd}/../videos_for_testing/{original_filename} --output {cwd}/detection_on_original_video/")
