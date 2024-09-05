import os
import base64
import numpy as np
import torch
import torchvision.transforms as T
import cv2
from flask import Flask, request, jsonify, render_template
import segmentation_models_pytorch as smp

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Model
def load_model():
    # model = smp.UnetPlusPlus('timm-resnest14d', encoder_weights=None, classes=1, activation=None, in_channels=3)
    model = smp.UnetPlusPlus('efficientnet-b0', encoder_weights=None, classes=1, activation=None, in_channels=3)
    model.load_state_dict(torch.load('Best_Dice_ver2.pt'))
    model.to(device)
    model.eval()
    return model

model = load_model()

mean = [0.154, 0.200, 0.154]
std = [0.061, 0.072, 0.058]

def preprocess_image(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def model_predict(image):
    image = image.to(device)

    with torch.no_grad():
        image = image.unsqueeze(0)
        output = model(image)
        output = torch.sigmoid(output)

        threshold = 0.5
        binary_mask = (output > threshold).float()
        pred_mask = binary_mask.cpu().squeeze(0).squeeze(0).numpy()

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
    
    # Pad the image to be divisible by tile_size
    padded_image = pad_to_nearest_multiple(image, tile_size)
    height, width = padded_image.shape[:2]
    
    # Calculate number of tiles
    num_tiles_x = width // tile_size
    num_tiles_y = height // tile_size

    # Initialize an empty mask
    full_mask = np.zeros((height, width), dtype=np.float32)

    for i in range(num_tiles_y):
        for j in range(num_tiles_x):
            y_start = i * tile_size
            x_start = j * tile_size
            y_end = y_start + tile_size
            x_end = x_start + tile_size

            # Extract tile
            tile = padded_image[y_start:y_end, x_start:x_end]
            tile_tensor = T.Compose([T.ToTensor(), T.Normalize(mean, std)])(tile)

            # Predict the tile
            pred_mask = model_predict(tile_tensor)

            # Ensure pred_mask has the same shape as the tile
            pred_mask_resized = np.zeros((tile_size, tile_size), dtype=np.float32)
            pred_mask_resized[:tile.shape[0], :tile.shape[1]] = pred_mask[:tile.shape[0], :tile.shape[1]]

            # Place the predicted mask in the correct location
            full_mask[y_start:y_end, x_start:x_end] = np.maximum(full_mask[y_start:y_end, x_start:x_end], pred_mask_resized[:tile.shape[0], :tile.shape[1]])

    return full_mask[:image.shape[0], :image.shape[1]]

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

    height, width = image.shape[:2]
    if height > 256 or width > 256:
        pred_mask = process_large_image(image)
    else:
        image = cv2.resize(image, (256, 256))
        image_tensor = T.Compose([T.ToTensor(), T.Normalize(mean, std)])(image)
        pred_mask = model_predict(image_tensor)

    seg = (pred_mask * 255).astype(np.uint8)

    main = image.copy()
    contours, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(main, contours, -1, (255, 255, 0), 4)

    fire_detected = bool(pred_mask.sum() > 0)

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

    height, width = image.shape[:2]
    if height > 256 or width > 256:
        pred_mask = process_large_image(image)
    else:
        image = cv2.resize(image, (256, 256))
        image_tensor = T.Compose([T.ToTensor(), T.Normalize(mean, std)])(image)
        pred_mask = model_predict(image_tensor)

    seg = (pred_mask * 255).astype(np.uint8)

    main = image.copy()
    contours, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(main, contours, -1, (255, 255, 0), 4)

    fire_detected = bool(pred_mask.sum() > 0)

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