from flask import Flask, render_template, request, Response, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
import os
import uuid
import cv2
import tensorflow as tf
import numpy as np
import time
from onvif import ONVIFCamera
# from . import detections

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'app/static/uploads'
app.config['RESULT_FOLDER'] = 'app/static/output'
cctv_streams = {}

detect_fn = tf.saved_model.load('static/HAR_model/saved_model')
active_stream = None
camera_active = False

class_names = {
    1: 'berdiri',
    2: 'berjalan',
    3: 'berkelahi',
    4: 'berlari',
    5: 'duduk',
    6: 'mencuri',
}

class_colors = {
    1: (255, 100, 100),  # Pink
    2: (0, 255, 0),    # Green
    3: (255, 0, 255),  # Purple
    4: (0, 255, 255),  # Yellow
    5: (255, 255, 0),  # Cyan
    6: (0, 0, 255),    # Red
}

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
        classes = detect_image(upload_path, result_path)

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

        classes = detect_video(upload_path, result_path)

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
        return Response(detect_realtime(uri, cctv=True),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        print("Local camera stream requested")
        return Response(detect_realtime(index, cctv=False),
                         mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_camera')
def stop_camera():
    print('trying to stop camera..')
    global active_stream, camera_active
    camera_active = False
    if active_stream:
        active_stream.release()
        active_stream = None
    print('camera stopped')
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

def detect_objects(image_np: cv2.typing.MatLike, input_tensor: tf.Tensor, show_action: bool = False):
    detections = detect_fn(input_tensor)

    # Ambil output
    boxes = detections['detection_boxes'][0].numpy()
    classes = detections['detection_classes'][0].numpy().astype(np.int32)
    scores = detections['detection_scores'][0].numpy()

    class_list = set()
    for i in range(len(scores)):
        if scores[i] >= 0.5:
            y1, x1, y2, x2 = boxes[i]
            p1 = (int(x1 * 500), int(y1 * 500))
            p2 = (int(x2 * 500), int(y2 * 500))

            class_list.add(class_names[classes[i]])
            cv2.rectangle(image_np, p1, p2, (0,0,0), 4)
            cv2.rectangle(image_np, p1, p2, class_colors[classes[i]], 2)
            if show_action:
                cv2.putText(image_np, class_names[classes[i]], (p1[0], p1[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)
                cv2.putText(image_np, class_names[classes[i]], (p1[0], p1[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, class_colors[classes[i]], 1)
    return image_np, class_list

def detect_image(image_path: str, result_path: str):
    image_np = cv2.imread(image_path)
    image_np = resize_with_padding(image_np)
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    input_tensor = tf.convert_to_tensor(image_rgb)[tf.newaxis, ...]

    image_np, classes = detect_objects( image_np, input_tensor)
    cv2.imwrite(result_path, image_np)

    return classes

def detect_video(video_path: str, result_path: str):
    cap = cv2.VideoCapture(video_path)
    # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(result_path, fourcc, fps, (500, 500))
    classes = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = resize_with_padding(frame)
        # Deteksi
        input_tensor = tf.convert_to_tensor(frame)[tf.newaxis, ...]

        if (frame.shape != input_tensor.shape[1:]):
            print("Frame shape does not match input tensor shape.")
        image_np, actions = detect_objects(frame, input_tensor, show_action=True)
        classes.update(actions)

        if image_np.dtype != np.uint8 or len(image_np.shape) != 3 or image_np.shape[2] != 3:
            image_np = cv2.convertScaleAbs(image_np)  # konversi agar aman ditulis

        out.write(image_np) 
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("kelas yang terdeteksi:",classes)
    return classes

def detect_realtime(uri: str, cctv: bool = False):
    target_fps = 20
    delay = 1.0 / target_fps
    if not cctv:
        uri = int(uri)
    cap = cv2.VideoCapture(uri)
    global active_stream, camera_active
    active_stream = cap
    camera_active = True
    try:
        while camera_active:
            start_time = time.time()
            success, frame = cap.read()
            if not success:
                break
            frame = resize_with_padding(frame)
            # Deteksi
            input_tensor = tf.convert_to_tensor(frame)[tf.newaxis, ...]

            # if (frame.shape != input_tensor.shape[1:]):
            #     print("Frame shape does not match input tensor shape.")
            image_np, _ = detect_objects(frame, input_tensor, show_action=True)
            _, buffer = cv2.imencode('.jpg', image_np)
            frame = buffer.tobytes()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            elapsed_time = time.time() - start_time
            if elapsed_time < delay:
                time.sleep(delay - elapsed_time)
    finally:
        print('releasing cv2 camera')
        cap.release()
        print('camera stopped')

def resize_with_padding(image: np.ndarray, color=(0, 0, 0)):
    h, w = image.shape[:2]

    # Hitung rasio skala agar proporsional
    scale = min(500 / w, 500 / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize gambar ke ukuran baru
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Buat canvas hitam dengan ukuran target
    padded_image = np.full((500, 500, 3), color, dtype=np.uint8)

    # Hitung posisi tengah untuk menempelkan gambar yang sudah di-resize
    x_offset = (500 - new_w) // 2
    y_offset = (500 - new_h) // 2

    # Tempelkan gambar ke canvas hitam
    padded_image[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_image

    return padded_image

