# Developed and maintained by: [ali fande]

import os, io, base64, numpy as np
from flask import Flask, request, jsonify, render_template
from PIL import Image, ImageOps
import torch, torch.nn as nn, torch.nn.functional as F

app = Flask(__name__)

class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.25)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

device = torch.device('cpu')
model = DigitCNN().to(device)
model_path = 'models/digit_model.pth'
model_ready = False

if os.path.exists(model_path):
    try:
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        model.eval()
        model_ready = True
        print('Model loaded successfully.')
    except Exception as e:
        print(f'Failed to load model: {e}')
else:
    print('WARNING: Model file not found.')


def preprocess_image(img):
    img = img.convert('L')
    img = ImageOps.invert(img)
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)
    img = img.resize((20, 20), Image.Resampling.LANCZOS)
    canvas = Image.new('L', (28, 28), 0)
    offset = ((28 - 20) // 2, (28 - 20) // 2)
    canvas.paste(img, offset)
    arr = np.array(canvas).astype('float32') / 255.0
    arr = (arr - 0.1307) / 0.3081
    arr = torch.tensor(arr).unsqueeze(0).unsqueeze(0).to(device)
    return arr


def predict_single(img):
    img_t = preprocess_image(img)
    with torch.no_grad():
        out = model(img_t)
        probs = F.softmax(out, dim=1)[0]
        conf, pred = torch.max(probs, 0)
    return int(pred.item()), float(conf.item()) * 100, probs.tolist()


def split_digits(img):
    gray = np.array(img.convert('L'))
    binary = (gray < 200).astype(np.uint8)
    col_sums = binary.sum(axis=0)
    in_digit, starts, ends = False, [], []
    for i, s in enumerate(col_sums):
        if not in_digit and s > 0:
            in_digit = True
            starts.append(i)
        elif in_digit and s == 0:
            in_digit = False
            ends.append(i)
    if in_digit:
        ends.append(len(col_sums))
    crops = []
    for s, e in zip(starts, ends):
        if e - s >= 3:
            crops.append(img.crop((max(0, s - 3), 0, min(img.width, e + 3), img.height)))
    return crops

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'model_ready': model_ready})

@app.route('/predict', methods=['POST'])
def predict():
    if not model_ready:
        return jsonify({'error': 'Model is not available. Train it first during build or upload a valid model file.'}), 503
    data = request.get_json(silent=True)
    if not data or 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400
    try:
        img_data = base64.b64decode(data['image'].split(',')[-1])
        img = Image.open(io.BytesIO(img_data))
        mode = data.get('mode', 'single')
        if mode == 'multi':
            crops = split_digits(img)
            if not crops:
                digit, conf, probs = predict_single(img)
                return jsonify({'mode': 'single', 'digit': digit, 'confidence': round(conf, 2), 'all_probs': [round(p * 100, 2) for p in probs]})
            results = []
            for crop in crops:
                d, c, _ = predict_single(crop)
                results.append({'digit': d, 'confidence': round(c, 2)})
            return jsonify({'mode': 'multi', 'results': results, 'number': ''.join(str(r['digit']) for r in results)})
        digit, conf, probs = predict_single(img)
        return jsonify({'mode': 'single', 'digit': digit, 'confidence': round(conf, 2), 'all_probs': [round(p * 100, 2) for p in probs]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
