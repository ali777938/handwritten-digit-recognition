import os, io, base64, numpy as np
from flask import Flask, request, jsonify, render_template
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import transforms

app = Flask(__name__)

class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout1(x)
        x = x.view(-1, 128 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        return self.fc2(x)

device = torch.device("cpu")
model = DigitCNN().to(device)
model_path = "models/digit_model.pth"

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model loaded.")
else:
    print("WARNING: Model not found. Run train_model.py first.")

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def preprocess_image(img):
    img = img.convert("RGBA")
    bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    bg.paste(img, mask=img.split()[3])
    img = bg.convert("RGB")
    return transform(img).unsqueeze(0).to(device)

def predict_single(img):
    img_t = preprocess_image(img)
    with torch.no_grad():
        out = model(img_t)
        probs = F.softmax(out, dim=1)[0]
        conf, pred = torch.max(probs, 0)
    return int(pred.item()), float(conf.item()) * 100, probs.tolist()

def split_digits(img):
    gray = np.array(img.convert("L"))
    binary = (gray < 128).astype(np.uint8)
    col_sums = binary.sum(axis=0)
    in_digit, starts, ends = False, [], []
    for i, s in enumerate(col_sums):
        if not in_digit and s > 0:
            in_digit = True; starts.append(i)
        elif in_digit and s == 0:
            in_digit = False; ends.append(i)
    if in_digit: ends.append(len(col_sums))
    crops = []
    for s, e in zip(starts, ends):
        if e - s >= 3:
            crops.append(img.crop((max(0, s-2), 0, min(img.width, e+2), img.height)))
    return crops

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

@app.route("/predict", methods=["POST"])
def predict():
    if not os.path.exists(model_path):
        return jsonify({"error": "Model not trained yet. Run train_model.py"}), 503
    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "No image provided"}), 400
    try:
        img_data = base64.b64decode(data["image"].split(",")[-1])
        img = Image.open(io.BytesIO(img_data))
        mode = data.get("mode", "single")
        if mode == "multi":
            crops = split_digits(img)
            if not crops:
                digit, conf, probs = predict_single(img)
                return jsonify({"mode": "single", "digit": digit, "confidence": round(conf,2), "all_probs": [round(p*100,2) for p in probs]})
            results = []
            for crop in crops:
                d, c, _ = predict_single(crop)
                results.append({"digit": d, "confidence": round(c,2)})
            return jsonify({"mode": "multi", "results": results, "number": "".join(str(r["digit"]) for r in results)})
        else:
            digit, conf, probs = predict_single(img)
            return jsonify({"mode": "single", "digit": digit, "confidence": round(conf,2), "all_probs": [round(p*100,2) for p in probs]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
