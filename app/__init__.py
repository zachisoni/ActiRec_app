from flask import Flask, render_template, request, Response, jsonify
from werkzeug.utils import secure_filename
import os
import uuid
import cv2
from . import detections

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'app/static/uploads'
app.config['RESULT_FOLDER'] = 'app/static/output'

@app.route('/')
def home():
    return render_template('home.html', page_title='Beranda')


@app.route('/image', methods=['GET', 'POST'])
def detect_image():
    if request.method == 'GET':
        return render_template('image.html', page_title='Gambar')
    file = request.files['file']
    print("File received:", file)
    if file:
        filename = secure_filename(file.filename)
        uid = str(uuid.uuid4())
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], 'images', uid + '_' + filename)
        result_path = os.path.join(app.config['RESULT_FOLDER'], 'images', uid + '_detected_' + filename)
        file.save(upload_path)

        # Deteksi objek
        classes = detections.detect_image(upload_path, result_path)

        return render_template('image.html', 
                                original_image=upload_path, 
                                detected_image=result_path, 
                                classes=classes,
                                page_title='Gambar')

@app.route('/video', methods=['GET', 'POST'])
def detect_video():
    if request.method == 'GET':
        return render_template('video.html', page_title='Video')
    file = request.files['file']
    if file:
        filename = secure_filename(file.filename)
        uid = str(uuid.uuid4())
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], 'videos', uid + '_' + filename)
        result_path = os.path.join(app.config['RESULT_FOLDER'], 'videos', uid + '_result_' + filename)

        file.save(upload_path)

        classes = detections.detect_video(upload_path, result_path)

        return render_template('video.html',
                                original_video=upload_path,
                                detected_video=result_path,
                                classes=classes,
                                page_title='Video')

@app.route('/realtime', methods=['GET', 'POST'])
def detect_realtime():
    return render_template('realtime.html', page_title='Realtime')

cameras = {}

def get_available_cameras(max_index=6):
    available = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.read()[0]:
            available.append({'index': i, 'name': f'Kamera {i}'})
        cap.release()
    return available

@app.route('/list_cameras')
def list_cameras():
    return jsonify({'cameras': get_available_cameras()})

@app.route('/video_feed/<int:index>')
def video_feed(index):
    return Response(detections.detect_realtime(index),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_camera')
def stop_camera():
    detections.stop_camera()
    return "Camera Stopped"
