from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
import mediapipe as mp

# Tamil character mapping
TAMIL_MAP = {
    0: ('1', 'அ'),
    1: ('117', 'ந'),
    2: ('119', 'நீ'),
    3: ('140', 'ம'),
    4: ('141', 'மா'),
    5: ('184', 'லை'),
    6: ('191', 'வீ'),
    7: ('22', 'ப்'),
    8: ('23', 'ம்'),
    9: ('25', 'ர்'),
    10: ('26', 'ல்'),
    11: ('29', 'ள்'),
    12: ('33', 'கா'),
    13: ('36', 'கு'),
    14: ('56', 'ச'),
    15: ('57', 'சா'),
    16: ('84', 'டு')
}

class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

app = Flask(__name__)
CORS(app)

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = CNNModel(num_classes=17).to(device)
model_path = r"D:\sign-language-trainer\backend\flask\model\pytorch_cnn_tamil_sign_17.pth"

# Verify model file exists
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model file not found at {model_path}")

try:
    # Load model with proper error handling
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"✅ Model loaded successfully from {model_path}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    raise

# Image transformation
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

def detect_and_crop_hand(image):
    # Convert image to RGB for MediaPipe
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image
    results = hands.process(image_rgb)
    
    # If no hands detected, return original image
    if not results.multi_hand_landmarks:
        return image, False
    
    # Get the first detected hand
    hand_landmarks = results.multi_hand_landmarks[0]
    
    # Find the bounding box of the hand
    h, w, _ = image.shape
    x_min, y_min = w, h
    x_max, y_max = 0, 0
    
    for landmark in hand_landmarks.landmark:
        x, y = int(landmark.x * w), int(landmark.y * h)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x)
        y_max = max(y_max, y)
    
    # Add padding to bounding box
    padding = 30
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(w, x_max + padding)
    y_max = min(h, y_max + padding)
    
    # Ensure minimum size of the bounding box
    bbox_w = x_max - x_min
    bbox_h = y_max - y_min
    min_size = 100
    
    if bbox_w < min_size:
        center_x = (x_min + x_max) // 2
        x_min = max(0, center_x - min_size // 2)
        x_max = min(w, center_x + min_size // 2)
    
    if bbox_h < min_size:
        center_y = (y_min + y_max) // 2
        y_min = max(0, center_y - min_size // 2)
        y_max = min(h, center_y + min_size // 2)
    
    # Crop the image to the hand bounding box
    cropped_image = image[y_min:y_max, x_min:x_max]
    
    # Draw the hand landmarks on the image (for debugging)
    # debug_image = image.copy()
    # mp_drawing.draw_landmarks(debug_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    # cv2.imwrite('debug_hand.jpg', debug_image)
    
    return cropped_image, True

def preprocess_image(image):
    # Detect and crop hand
    hand_image, hand_detected = detect_and_crop_hand(image)
    
    if not hand_detected:
        return None, False
    
    # Convert to RGB
    hand_image = cv2.cvtColor(hand_image, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image
    pil_image = Image.fromarray(hand_image)
    
    # Apply transformations
    tensor = transform(pil_image)
    
    # Add batch dimension
    tensor = tensor.unsqueeze(0).to(device)
    return tensor, True

@app.route("/process-frame", methods=["POST"])
def process_frame():
    try:
        # Get image file from request
        file = request.files["frame"]
        
        # Read image
        img_array = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({"error": "Invalid image data"}), 400
        
        # Preprocess image (detect and crop hand, then transform)
        tensor, hand_detected = preprocess_image(frame)
        
        if not hand_detected:
            return jsonify({"letter": None, "confidence": 0.0, "handDetected": False})
        
        # Get prediction
        with torch.no_grad():
            outputs = model(tensor)
            probabilities = torch.softmax(outputs, dim=1)
            max_prob, predicted = torch.max(probabilities, 1)
            
            confidence = float(max_prob.item())
            if confidence > 0.7:  # Confidence threshold
                predicted_idx = predicted.item()
                if predicted_idx in TAMIL_MAP:
                    folder, tamil = TAMIL_MAP[predicted_idx]
                    print(f"✅ Detected sign: {tamil} (confidence: {confidence:.2f})")
                    return jsonify({
                        "letter": tamil,
                        "confidence": confidence,
                        "folder": folder,
                        "handDetected": True
                    })
        
        return jsonify({"letter": None, "confidence": 0.0, "handDetected": True})

    except Exception as e:
        print(f"❌ Error processing frame: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("✅ Flask server starting...")
    app.run(host="0.0.0.0", port=5000, debug=True)
