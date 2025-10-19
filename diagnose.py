import os
import numpy as np
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
CORS(app)

# Load the skin classification model
try:
    model = load_model('dermalscan_resnet_model.h5')
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

@app.route('/diagnose', methods=['POST'])
def diagnose():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Save uploaded file temporarily
        filename = file.filename
        filepath = os.path.join('uploads', filename)
        os.makedirs('uploads', exist_ok=True)
        file.save(filepath)
        
        # Preprocess image
        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        
        # Make prediction
        predictions = model.predict(img_array)
        class_index = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        
        # Class names
        class_names = ['clear_face', 'dark_spots', 'puffy_eyes', 'wrinkles']
        predicted_class = class_names[class_index]
        
        # Clean up temporary file
        os.remove(filepath)
        
        return jsonify({
            'prediction': predicted_class,
            'confidence': confidence,
            'class_probabilities': {
                class_names[i]: float(predictions[0][i]) 
                for i in range(len(class_names))
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)
