# app.py
import os
from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mkv'}  # Define allowed video formats


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


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

        # Process the video and detect guns
        processed_video_path = process_video(video_path)

        return redirect(url_for('show_result', video_name=video_file.filename))

    return redirect(url_for('index'))


def process_video(video_path):
    # TODO: Implement video processing code from the previous section

    # For now, just return the original video path as a placeholder
    return video_path


@app.route('/result/<video_name>')
def show_result(video_name):
    processed_video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_name)

    return render_template('result.html', video_name=video_name, processed_video_path=processed_video_path)


if __name__ == '__main__':
    app.run(debug=True)
