import os
import numpy as np
import cv2 # OpenCV for image processing
import requests 
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


# --- Initialization ---
app = Flask(__name__)
CORS(app)

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
ANNOTATED_FOLDER = 'annotated'
MODELS_FOLDER = 'models' 
for folder in [UPLOAD_FOLDER, ANNOTATED_FOLDER, MODELS_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ANNOTATED_FOLDER'] = ANNOTATED_FOLDER

# --- Load Skin Feature Classification Model ---
try:
    # --- THIS IS THE LINE TO CHANGE ---
    # We are now loading the new, more powerful ResNet model
    CLASSIFICATION_MODEL_PATH = 'dermalscan_resnet_model.h5' 
    classification_model = load_model(CLASSIFICATION_MODEL_PATH)
    print(f"✅ Skin feature model loaded successfully from {CLASSIFICATION_MODEL_PATH}")
    DATASET_PATH = 'dataset'
    if os.path.exists(DATASET_PATH):
        class_names = sorted([d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))])
        print(f"✅ Class names loaded: {class_names}")
    else:
        print(f"⚠️ Warning: Dataset path not found.")
        class_names = ['clear_face', 'dark_spots', 'puffy_eyes', 'wrinkles']
except Exception as e:
    print(f"❌ Error loading skin feature model: {e}")
    classification_model = None
    class_names = []

# --- Load Age Prediction Model ---
AGE_MODEL_PROTO = os.path.join(MODELS_FOLDER, 'age_deploy.prototxt')
AGE_MODEL_WEIGHTS = os.path.join(MODELS_FOLDER, 'age_net.caffemodel')
AGE_VALUES = [1.5, 5, 10.5, 17.5, 28.5, 40.5, 50.5, 70]
age_net = None
try:
    age_net = cv2.dnn.readNet(AGE_MODEL_PROTO, AGE_MODEL_WEIGHTS)
    print("✅ Age prediction model loaded successfully from local files.")
except Exception as e:
    print(f"❌ Could not load age prediction model: {e}")

# --- Load DNN Face Detector Model ---
FACE_MODEL_PROTO = os.path.join(MODELS_FOLDER, 'deploy.prototxt')
FACE_MODEL_WEIGHTS = os.path.join(MODELS_FOLDER, 'res10_300x300_ssd_iter_140000.caffemodel')
face_net = None
try:
    face_net = cv2.dnn.readNet(FACE_MODEL_PROTO, FACE_MODEL_WEIGHTS)
    print("✅ DNN Face detector loaded successfully from local files.")
except Exception as e:
    print(f"❌ Could not load face detector model: {e}")


# --- Helper Functions ---
def predict_age(face_blob):
    """Predicts a specific age using a weighted average of the top 2 model outputs."""
    if age_net is None:
        return "Not Estimated"
    age_net.setInput(face_blob)
    preds = age_net.forward()[0]
    top_indices = preds.argsort()[-2:][::-1]
    top1_conf, top2_conf = preds[top_indices[0]], preds[top_indices[1]]
    top1_age, top2_age = AGE_VALUES[top_indices[0]], AGE_VALUES[top_indices[1]]
    if (top1_conf + top2_conf) > 0:
        estimated_age = (top1_age * top1_conf + top2_age * top2_conf) / (top1_conf + top2_conf)
    else:
        estimated_age = top1_age
    return f"~{int(round(estimated_age))} years"

def analyze_skin(image_path):
    """Predicts skin features using the Keras model."""
    if classification_model is None: return {"error": "Skin model not loaded"}
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0) / 255.0
    predictions = classification_model.predict(img_array_expanded)
    class_index = np.argmax(predictions[0])
    return {
        "label": class_names[class_index].replace('_', ' ').title(),
        "confidence": float(np.max(predictions[0]))
    }

# --- API Endpoints ---
@app.route('/analyze', methods=['POST'])
def analyze_image_endpoint():
    if 'file' not in request.files: return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '': return jsonify({'error': 'No selected file'}), 400

    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        skin_analysis_result = analyze_skin(filepath)
        if "error" in skin_analysis_result: return jsonify(skin_analysis_result), 500

        # --- Using the more accurate DNN Face Detector ---
        img = cv2.imread(filepath)
        (h, w) = img.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        
        age_prediction = "Face Not Detected"
        if face_net:
            face_net.setInput(blob)
            detections = face_net.forward()
            best_face_index = np.argmax(detections[0, 0, :, 2])
            confidence = detections[0, 0, best_face_index, 2]

            if confidence > 0.5:
                box = detections[0, 0, best_face_index, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                # --- Add padding to the face crop for better age prediction ---
                padding = 20
                padded_startX = max(0, startX - padding)
                padded_startY = max(0, startY - padding)
                padded_endX = min(w, endX + padding)
                padded_endY = min(h, endY + padding)

                face = img[padded_startY:padded_endY, padded_startX:padded_endX]
                
                if face.size != 0:
                    face_blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
                    age_prediction = predict_age(face_blob)
                    # Draw the original (non-padded) rectangle for a cleaner look
                    cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 2)

        # --- Annotate Image ---
        annotated_filename = os.path.basename(filepath)
        annotated_path = os.path.join(app.config['ANNOTATED_FOLDER'], annotated_filename)
        skin_label = f"{skin_analysis_result['label']}: {skin_analysis_result['confidence']:.2%}"
        age_label = f"Age: {age_prediction}"
        cv2.putText(img, skin_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, skin_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, age_label, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, age_label, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(annotated_path, img)

        return jsonify({
            "age_prediction": age_prediction,
            "detected_features": [skin_analysis_result],
            "annotated_image_url": f"http://127.0.0.1:5000/annotated/{annotated_filename}"
        })

@app.route('/annotated/<filename>')
def send_annotated_file(filename):
    return send_from_directory(app.config['ANNOTATED_FOLDER'], filename)

@app.route('/')
def index():
    return "DermalScan AI Backend is running!"

if __name__ == '__main__':
    app.run(debug=False, port=5000)
