from flask import Flask, request, jsonify, render_template
import os
from werkzeug.utils import secure_filename
import numpy as np
from cnn_model import train_model, predict_image
from PIL import Image
import shutil

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
MODEL_FOLDER = "model"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MODEL_FOLDER"] = MODEL_FOLDER

class_images = {}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_images():
    class_names = request.form.getlist("class_names[]")

    if not class_names:
        return jsonify({"error": "At least one class is required"}), 400

    for i, class_name in enumerate(class_names):
        class_path = os.path.join(UPLOAD_FOLDER, class_name)
        os.makedirs(class_path, exist_ok=True)
        
        uploaded_files = request.files.getlist(f"images_{i}[]")
        if not uploaded_files:
            return jsonify({"error": f"No images uploaded for class {class_name}"}), 400
        
        for file in uploaded_files:
            filename = secure_filename(file.filename)
            file_path = os.path.join(class_path, filename)
            file.save(file_path)

    return jsonify({"message": "Images uploaded successfully", "classes": class_names})

@app.route("/train", methods=["POST"])
def train():
    dataset_path = "static/uploads"
    model_path = os.path.join(MODEL_FOLDER, "cnn_model.h5")

    result = train_model(dataset_path, model_path)

    if "error" in result:
        return jsonify({"error": result["error"]})

    return jsonify({
        "message": "Model trained successfully!",
        "accuracy": result["accuracy"]  # Ensure it's a valid number
    })



@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)  # Save the image temporarily for display

    model_path = os.path.join(MODEL_FOLDER, "cnn_model.h5")
    predicted_class = predict_image(Image.open(file).convert("RGB"), model_path)

    return jsonify({
        "predicted_class": predicted_class,
        "image_url": f"/static/uploads/{filename}"
    })


@app.route("/clear_uploads", methods=["POST"])
def clear_uploads():
    shutil.rmtree(UPLOAD_FOLDER)  # Delete all images
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Recreate empty folder
    return jsonify({"message": "Uploads cleared successfully!"})

@app.route("/clear_model", methods=["POST"])
def clear_model():
    model_path = os.path.join(MODEL_FOLDER, "cnn_model.h5")
    class_indices_path = model_path.replace(".h5", "_class_indices.json")

    # Check and delete model file
    if os.path.exists(model_path):
        os.remove(model_path)

    # Check and delete class indices file
    if os.path.exists(class_indices_path):
        os.remove(class_indices_path)

    return jsonify({"message": "Model cleared successfully!"})

if __name__ == "__main__":
    app.run(debug=True)
