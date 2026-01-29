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

# ===================== CLASS NAMES (DO NOT CHANGE ORDER) =====================
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
        "cure": "No treatment required. Maintain proper irrigation, nutrition, and monitoring."
    },
    "Bacterial Spot": {
        "description": "A bacterial disease causing dark, water-soaked spots on leaves.",
        "cure": "Remove infected leaves and apply copper-based bactericides."
    },
    "Black Rot": {
        "description": "A fungal disease that causes dark circular lesions and leaf decay.",
        "cure": "Prune affected areas and apply appropriate fungicides."
    },
    "Early Blight": {
        "description": "Characterized by brown spots with concentric rings on older leaves.",
        "cure": "Apply fungicides and practice crop rotation."
    },
    "Late Blight": {
        "description": "A serious disease causing rapid leaf browning and decay.",
        "cure": "Use certified seeds and apply systemic fungicides immediately."
    },
    "Leaf Blight": {
        "description": "Causes elongated brown lesions that lead to leaf drying.",
        "cure": "Remove infected foliage and improve field drainage."
    },
    "Leaf Mold": {
        "description": "Fungal infection producing yellow spots and mold growth under leaves.",
        "cure": "Ensure good air circulation and apply fungicides."
    },
    "Leaf Scab": {
        "description": "Produces scabby lesions and deformation on leaves.",
        "cure": "Use resistant varieties and apply protective fungicides."
    },
    "Leaf Scorch": {
        "description": "Leaf edges dry out and turn brown due to stress or infection.",
        "cure": "Maintain proper watering and remove infected leaves."
    },
    "Northern Leaf Blight": {
        "description": "Forms long grayish lesions on leaves, reducing photosynthesis.",
        "cure": "Apply fungicides and practice crop rotation."
    },
    "Powdery Mildew": {
        "description": "A fungal disease forming white powdery patches on leaves.",
        "cure": "Apply sulfur-based fungicides and reduce humidity."
    },
    "Rust": {
        "description": "Produces reddish-brown pustules on leaf surfaces.",
        "cure": "Remove infected leaves and apply appropriate fungicides."
    },
    "Septoria Leaf Spot": {
        "description": "Small dark spots with light centers appear on leaves.",
        "cure": "Remove affected leaves and apply fungicides."
    },
    "Target Spot": {
        "description": "Circular lesions with concentric rings appear on leaves.",
        "cure": "Use fungicides and avoid overhead irrigation."
    },
    "Mosaic Virus": {
        "description": "Viral disease causing mottled leaf coloration and distortion.",
        "cure": "Remove infected plants and control insect vectors."
    },
    "Esca Black Measles": {
        "description": "Causes dark streaks and spots leading to vine decline.",
        "cure": "Prune infected vines and apply disease management practices."
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

    img = load_img(filepath, target_size=(224, 224))
    x = img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)

    preds = model.predict(x)[0]
    idx = int(np.argmax(preds))
    confidence = round(float(preds[idx]) * 100, 2)

    label = CLASS_NAMES[idx]
    info = disease_info[label]

    return render_template(
        "result.html",
        prediction=label,
        confidence=confidence,
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
