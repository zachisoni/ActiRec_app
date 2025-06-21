from flask import Flask, render_template, request, Response, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
import os
import uuid
import cv2
from onvif import ONVIFCamera
from . import detections

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'app/static/uploads'
app.config['RESULT_FOLDER'] = 'app/static/output'
cctv_streams = {}

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

@app.route('/video_feed/<int:index>/<int:cctv>')
def video_feed(index, cctv: bool):
    if cctv:
        print("CCTV stream requested")
        uri = cctv_streams.get(index)
        if not uri:
            return "Kamera tidak ditemukan", 404
        return Response(detections.detect_realtime(uri, cctv=True),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        print("Local camera stream requested")
        return Response(detections.detect_realtime(index, cctv=False),
                         mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_camera')
def stop_camera():
    detections.stop_camera()
    return "Camera Stopped"

@app.route('/connect_cctv', methods=['POST'])
def connect_camera():
    ip = request.form['ip']
    port = int(request.form['port'])
    username = request.form['username']
    password = request.form['password']
    
    uri = get_rtsp_uri(ip, port, username, password)
    if uri:
        index = len(cctv_streams)
        cctv_streams[index] = uri
        return redirect(url_for('video_feed', index=index, cctv=True))
    else:
        return "Gagal terhubung ke kamera", 400

def get_rtsp_uri(ip, port, username, password):
    try:
        cam = ONVIFCamera(ip, port, username, password)
        media_service = cam.create_media_service()
        profiles = media_service.GetProfiles()
        token = profiles[0].token
        stream_uri = media_service.GetStreamUri({
            'StreamSetup': {
                'Stream': 'RTP-Unicast',
                'Transport': {'Protocol': 'RTSP'}
            },
            'ProfileToken': token
        })
        return stream_uri.Uri
    except Exception as e:
        print("ONVIF connection failed:", e)
        return None