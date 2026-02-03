import os
import numpy as np
from flask import Flask, request, render_template, send_from_directory
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)

# ===================== UPLOAD CONFIG =====================
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ===================== LOAD MODEL =====================
model = load_model("model.h5")
print("✅ Model loaded. Visit http://127.0.0.1:5000/")

# ===================== CLASS NAMES (ORDER MUST MATCH TRAINING) =====================
CLASS_NAMES = {
    0: "Bacterial Spot",
    1: "Black Rot",
    2: "Early Blight",
    3: "Esca Black Measles",
    4: "Healthy",
    5: "Late Blight",
    6: "Leaf Blight",
    7: "Leaf Mold",
    8: "Leaf Scab",
    9: "Leaf Scorch",
    10: "Mosaic Virus",
    11: "Northern Leaf Blight",
    12: "Powdery Mildew",
    13: "Rust",
    14: "Septoria Leaf Spot",
    15: "Target Spot"
}

# ===================== DISEASE INFORMATION =====================
disease_info = {
    "Healthy": {
        "description": "The leaf shows no visible signs of disease and appears healthy.",
        "cure": "No treatment required. Maintain proper irrigation, nutrition, and regular monitoring."
    },
    "Bacterial Spot": {
        "description": "Dark, water-soaked spots caused by bacterial infection.",
        "cure": "Remove infected leaves and apply copper-based bactericides."
    },
    "Black Rot": {
        "description": "Fungal disease causing black circular lesions and decay.",
        "cure": "Prune affected parts and apply fungicides."
    },
    "Early Blight": {
        "description": "Brown spots with concentric rings, mostly on older leaves.",
        "cure": "Apply fungicides and practice crop rotation."
    },
    "Late Blight": {
        "description": "Rapid leaf browning and decay under moist conditions.",
        "cure": "Apply systemic fungicides and remove infected plants."
    },
    "Leaf Blight": {
        "description": "Elongated brown lesions that dry out leaves.",
        "cure": "Improve drainage and remove infected foliage."
    },
    "Leaf Mold": {
        "description": "Yellow patches with mold growth on leaf undersides.",
        "cure": "Ensure airflow and apply fungicides."
    },
    "Leaf Scab": {
        "description": "Scabby lesions causing deformation of leaves.",
        "cure": "Use resistant varieties and protective fungicides."
    },
    "Leaf Scorch": {
        "description": "Dry brown edges due to stress or infection.",
        "cure": "Maintain proper watering and remove damaged leaves."
    },
    "Northern Leaf Blight": {
        "description": "Long gray-green lesions reducing photosynthesis.",
        "cure": "Apply fungicides and rotate crops."
    },
    "Powdery Mildew": {
        "description": "White powdery fungal growth on leaf surface.",
        "cure": "Apply sulfur-based fungicides and reduce humidity."
    },
    "Rust": {
        "description": "Reddish-brown pustules on leaf surfaces.",
        "cure": "Remove infected leaves and apply fungicides."
    },
    "Septoria Leaf Spot": {
        "description": "Small dark spots with light centers.",
        "cure": "Remove infected leaves and apply fungicides."
    },
    "Target Spot": {
        "description": "Circular target-like lesions on leaves.",
        "cure": "Avoid overhead irrigation and apply fungicides."
    },
    "Mosaic Virus": {
        "description": "Mottled leaf coloration caused by viral infection.",
        "cure": "Remove infected plants and control insect vectors."
    },
    "Esca Black Measles": {
        "description": "Dark streaks and spots leading to vine decline.",
        "cure": "Prune infected vines and apply management practices."
    }
}

# ===================== ROUTES =====================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    # Image preprocessing
    img = load_img(filepath, target_size=(224, 224))
    x = img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)

    # Model prediction
    preds = model.predict(x)[0]

    # Sort predictions (Top-1 & Top-2)
    sorted_indices = np.argsort(preds)[::-1]
    top1_idx = sorted_indices[0]
    top2_idx = sorted_indices[1]

    top1_label = CLASS_NAMES[top1_idx]
    top2_label = CLASS_NAMES[top2_idx]

    top1_conf = round(float(preds[top1_idx]) * 100, 2)
    top2_conf = round(float(preds[top2_idx]) * 100, 2)

    # Confidence explanation
    if top1_conf >= 75:
        confidence_msg = "High confidence prediction."
    elif top1_conf >= 50:
        confidence_msg = "Moderate confidence. A clearer image may improve accuracy."
    else:
        confidence_msg = "Low confidence. Diseases may look similar or image quality may be low."

    info = disease_info[top1_label]

    return render_template(
        "result.html",
        prediction=top1_label,
        confidence=top1_conf,
        alt_prediction=top2_label,
        alt_confidence=top2_conf,
        confidence_msg=confidence_msg,
        description=info["description"],
        cure=info["cure"],
        image_path="/uploads/" + filename
    )

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# ===================== RUN =====================
if __name__ == "__main__":
    app.run(debug=True)

@app.route("/login")
def login():
    return render_template("login.html")

@app.route("/register")
def register():
    return render_template("register.html")
