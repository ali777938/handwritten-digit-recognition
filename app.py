import os
import io
import base64
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template
from PIL import Image, ImageOps

app = Flask(__name__)

model_path = 'models/digit_model.pkl'
model = None
model_ready = False

if os.path.exists(model_path):
    try:
        model = joblib.load(model_path)
        model_ready = True
        print('Model loaded successfully.')
    except Exception as e:
        print(f'Failed to load model: {e}')
else:
    print('WARNING: Model file not found.')


def center_digit(gray_img):
    arr = np.array(gray_img)
    ys, xs = np.where(arr > 0)
    if len(xs) == 0 or len(ys) == 0:
        return gray_img
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    cropped = gray_img.crop((x1, y1, x2 + 1, y2 + 1))
    w, h = cropped.size
    scale = min(6 / max(w, 1), 6 / max(h, 1))
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new('L', (8, 8), 0)
    paste_x = (8 - new_w) // 2
    paste_y = (8 - new_h) // 2
    canvas.paste(resized, (paste_x, paste_y))
    return canvas


def preprocess_image(img):
    img = img.convert('L')
    img = ImageOps.invert(img)
    img = ImageOps.autocontrast(img)
    img = center_digit(img)
    arr = np.array(img).astype(np.float32)
    arr = (arr / 255.0) * 16.0
    return arr.reshape(1, -1)


def predict_single(img):
    features = preprocess_image(img)
    probs = model.predict_proba(features)[0]
    pred = int(np.argmax(probs))
    conf = float(probs[pred] * 100)
    return pred, conf, probs.tolist()


def split_digits(img):
    gray = np.array(img.convert('L'))
    binary = (gray < 220).astype(np.uint8)
    col_sums = binary.sum(axis=0)
    in_digit = False
    starts, ends = [], []
    gap_count = 0
    for i, s in enumerate(col_sums):
        if not in_digit and s > 0:
            in_digit = True
            starts.append(i)
            gap_count = 0
        elif in_digit:
            if s == 0:
                gap_count += 1
                if gap_count >= 2:
                    in_digit = False
                    ends.append(i - gap_count + 1)
                    gap_count = 0
            else:
                gap_count = 0
    if in_digit:
        ends.append(len(col_sums))

    crops = []
    for s, e in zip(starts, ends):
        if e - s >= 2:
            crops.append(img.crop((max(0, s - 2), 0, min(img.width, e + 2), img.height)))
    return crops

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'model_ready': model_ready})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if not model_ready:
            return jsonify({'error': 'Model is not available yet.'}), 503
        data = request.get_json(silent=True)
        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400

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

@app.errorhandler(500)
def handle_500(_):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
