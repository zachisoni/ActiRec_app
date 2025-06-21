import cv2
import tensorflow as tf
import numpy as np
import time

detect_fn = tf.saved_model.load('app/static/HAR_model/saved_model')
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

@tf.function
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

def stop_camera():
    print('trying to stop camera..')
    global active_stream, camera_active
    camera_active = False
    if active_stream:
        active_stream.release()
        active_stream = None

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

