import os
import base64
import numpy as np
import onnxruntime
import cv2
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load Model
def load_model():
    sess_options = onnxruntime.SessionOptions()
    sess_options.intra_op_num_threads = 4  # Set the number of threads to use
    model = onnxruntime.InferenceSession("model_ver2.onnx", sess_options=sess_options, providers=["CPUExecutionProvider"])
    return model

model = load_model()

mean = [0.154, 0.199, 0.154]
std = [0.065, 0.075, 0.061]

def preprocess_image(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def preprocess_for_model(image):
    # Vectorize the normalization process
    img = (image.astype(np.float32) / 255.0 - mean) / std
    return img

def model_predict(image):
    ort_inputs = {model.get_inputs()[0].name: image}
    pred_mask = model.run(None, ort_inputs)
    return pred_mask

def pad_to_nearest_multiple(image, tile_size):
    height, width = image.shape[:2]
    new_height = ((height + tile_size - 1) // tile_size) * tile_size
    new_width = ((width + tile_size - 1) // tile_size) * tile_size
    
    # Pad the image
    padded_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)
    padded_image[:height, :width] = image
    return padded_image

def process_large_image(image):
    tile_size = 256
    padded_image = pad_to_nearest_multiple(image, tile_size)
    height, width = padded_image.shape[:2]
    num_tiles_x = width // tile_size
    num_tiles_y = height // tile_size

    full_mask = np.zeros((height, width), dtype=np.float32)

    for i in range(num_tiles_y):
        for j in range(num_tiles_x):
            y_start, x_start = i * tile_size, j * tile_size
            tile = preprocess_for_model(padded_image[y_start:y_start+tile_size, x_start:x_start+tile_size])
            tile_np = np.expand_dims(np.transpose(np.float32(tile), (2, 0, 1)), axis=0)
            
            pred_mask = np.squeeze(model_predict(tile_np))
            full_mask[y_start:y_start+tile_size, x_start:x_start+tile_size] = np.maximum(full_mask[y_start:y_start+tile_size, x_start:x_start+tile_size], pred_mask)

    return full_mask[:image.shape[0], :image.shape[1]]

def process_small_image(image):
    image = cv2.resize(image, (256, 256))
    img = preprocess_for_model(image)
    img = np.expand_dims(np.transpose(np.float32(img), (2, 0, 1)), axis=0)
    pred_mask = model_predict(img)

    return pred_mask, image

def detect_contours(seg):
    contours, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    img_bytes = file.read()
    image = preprocess_image(img_bytes)

    # processLargeImage 값을 클라이언트로부터 읽어옴
    process_large = request.form.get('processLargeImage', 'false').lower() == 'true'
    print(process_large)

    if process_large: pred_mask = process_large_image(image) if max(image.shape[:2]) > 256 else process_small_image(image)
    else: pred_mask, image = process_small_image(image)

    seg = np.uint8(np.squeeze(pred_mask) * 255)

    main = image.copy()
    contours = detect_contours(seg)
    cv2.drawContours(main, contours, -1, (255, 255, 0), 4)

    fire_detected = bool(np.sum(seg) > 0)

    def encode_image(img):
        _, buffer = cv2.imencode('.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        return base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        'input_image': encode_image(image),
        'mask_image': encode_image(seg),
        'contour_image': encode_image(main),
        'fire_detected': fire_detected
    })

@app.route('/predict_demo')
def predict_demo():
    image_name = request.args.get('image')
    if not image_name:
        return jsonify({'error': 'No image selected'}), 400

    image_path = os.path.join('static', 'images', image_name)
    with open(image_path, 'rb') as f:
        img_bytes = f.read()

    image = preprocess_image(img_bytes)

    # processLargeImage 값을 클라이언트로부터 읽어옴
    process_large = request.args.get('processLargeImage', 'false').lower() == 'true'

    if process_large: pred_mask = process_large_image(image) if max(image.shape[:2]) > 256 else process_small_image(image)
    else: pred_mask, image = process_small_image(image)

    seg = np.uint8(np.squeeze(pred_mask) * 255)

    main = image.copy()
    contours = detect_contours(seg)
    cv2.drawContours(main, contours, -1, (255, 255, 0), 4)

    fire_detected = bool(np.sum(seg) > 0)

    def encode_image(img):
        _, buffer = cv2.imencode('.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        return base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        'input_image': encode_image(image),
        'mask_image': encode_image(seg),
        'contour_image': encode_image(main),
        'fire_detected': fire_detected
    })

if __name__ == '__main__':
    app.run(debug=True)