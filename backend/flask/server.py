from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os

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

def preprocess_image(image):
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Convert to PIL Image
    pil_image = Image.fromarray(image)
    # Apply transformations
    tensor = transform(pil_image)
    # Add batch dimension
    tensor = tensor.unsqueeze(0).to(device)
    return tensor

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
        
        # Preprocess image
        tensor = preprocess_image(frame)
        
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
                        "folder": folder
                    })
        
        return jsonify({"letter": None, "confidence": 0.0})

    except Exception as e:
        print(f"❌ Error processing frame: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("✅ Flask server starting...")
    app.run(host="0.0.0.0", port=5000, debug=True)
