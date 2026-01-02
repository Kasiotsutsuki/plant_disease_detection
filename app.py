import os
import numpy as np
from flask import Flask, request, render_template, send_from_directory
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = load_model("model.h5")
print("Model loaded. Visit http://127.0.0.1:5000/")

labels = ["Healthy", "Powdery", "Rust"]

disease_info = {
    "Healthy": {
        "description": "The leaf appears healthy with no visible disease symptoms.",
        "cure": "No treatment required. Maintain proper watering and nutrition."
    },
    "Powdery": {
        "description": "Powdery mildew causes white powder-like fungal spots on leaves.",
        "cure": "Remove infected leaves and apply sulfur-based fungicides."
    },
    "Rust": {
        "description": "Rust disease forms orange or brown pustules on leaf surfaces.",
        "cure": "Apply fungicide, remove infected leaves, and improve air circulation."
    }
}

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
    idx = np.argmax(preds)
    confidence = round(float(preds[idx]) * 100, 2)

    label = labels[idx]
    info = disease_info[label]

    return render_template(
        "result.html",
        prediction=label,
        confidence=confidence,
        description=info["description"],
        cure=info["cure"],
        image_path="/uploads/" + filename
    )

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


if __name__ == "__main__":
    app.run(debug=True)
