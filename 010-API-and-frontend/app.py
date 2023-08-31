# app.py
import os
from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
from flask import send_from_directory
from tracking import process_video_with_detection
import json

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(app.root_path, 'static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
RESULTS_FOLDER = os.path.join(app.root_path, 'static', 'results')
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mkv'}  # Define allowed video formats


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def read_json_file(file_path):
    if not os.path.exists(file_path):
        return {}

    with open(file_path, 'r') as file:
        return json.load(file)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return redirect(url_for('index'))

    video_file = request.files['video']
    if video_file.filename == '':
        return redirect(url_for('index'))

    if video_file and allowed_file(video_file.filename):
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_file.filename)
        video_file.save(video_path)

        method = request.form.get('method')

        # Process the video and detect guns with the selected method
        processed_video_path = process_video(video_path, method)

        return redirect(url_for('show_result', video_name=video_file.filename))

    return redirect(url_for('index'))


def process_video(video_path, method):
    # TODO: Implement video processing code from the previous section
    process_video_with_detection(video_path, method=method)

    # For now, just return the original video path as a placeholder
    return video_path


@app.route('/static/results/<path:filename>')
def serve_video(filename):
    return send_from_directory(app.config['RESULTS_FOLDER'], filename, conditional=True)


@app.route('/result/<video_name>')
def show_result(video_name):
    data = read_json_file("data.json")
    processed_video_path = os.path.join(app.config['RESULTS_FOLDER'], video_name)
    return render_template('result.html', video_name=video_name, data=data, processed_video_path=processed_video_path)


if __name__ == '__main__':
    app.run(debug=True)
