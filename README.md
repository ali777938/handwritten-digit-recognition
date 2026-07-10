# Handwritten Digit Recognition Web Application

A deep learning web application that recognizes handwritten digits (single or multi-digit) using a CNN trained on MNIST.

## Group Members
- Ali_324349
- farouk_331844
- Emad_341459
- Ahmad_341668
- Ahmad_341814

## Features
- Single digit recognition with confidence scores and probability for all 10 digits
- Multi-digit recognition (draw "123" and get each digit recognized)
- Touch support for mobile/tablet
- Image upload support
- Light/dark mode

## Tech Stack
- **Backend**: Python, Flask, PyTorch
- **Model**: CNN trained on MNIST (~99% accuracy)
- **Frontend**: HTML5 Canvas, CSS3, Vanilla JavaScript

## Local Setup
```bash
pip install -r requirements.txt
python train_model.py    # Train the model
python app.py            # Start the server
# Open http://localhost:5000
```

## Deploy on Render.com
1. Push this repo to GitHub
2. Go to render.com → New + → Web Service
3. Connect your GitHub repository
4. Render auto-detects render.yaml
   - Build: `pip install -r requirements.txt && python train_model.py`
   - Start: `gunicorn app:app`

## Model Architecture
- 3x Conv layers (32→64→128 filters) + MaxPool
- Dropout (0.25, 0.5) for regularization
- FC layers: 256 → 10 classes
- Adam optimizer, 10 epochs, ~99% test accuracy

## Multi-Digit Detection Logic
1. Convert image to grayscale binary
2. Scan columns for connected regions of ink
3. Crop each digit individually
4. Run CNN prediction on each crop
5. Combine results into the full number


