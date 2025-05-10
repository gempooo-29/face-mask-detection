from flask import Flask, render_template, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
import io
import base64
from PIL import Image

# Initialize the Flask app and load the pre-trained model
app = Flask(__name__)
model = load_model('mask_classifier_model.keras')  # Path to your model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Route to render the upload form
@app.route('/')
def home():
    return render_template('index.html')  # Render the main page with the upload form

# Route to handle image prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', prediction="No file uploaded")
    
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', prediction="No file selected")

    # Convert the uploaded image to a format suitable for prediction
    img_bytes = file.read()
    npimg = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if img is None:
        return render_template('index.html', prediction="Failed to read the image")

    # Convert the image to grayscale and detect faces
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return render_template('index.html', prediction="No faces detected")

    # Loop through detected faces
    for (x, y, w, h) in faces:
        face = img[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (224, 224))  # Resize to match model input
        face_array = img_to_array(face_resized)
        face_array = np.expand_dims(face_array, axis=0)  # Add batch dimension
        face_array = face_array / 255.0  # Normalize the image

        # Get the prediction
        pred = model.predict(face_array)[0][0]
        label = "Mask" if pred < 0.5 else "No Mask"

        # Draw the bounding box and label on the image
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)  # Green for Mask, Red for No Mask
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)  # Draw the bounding box
        cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)  # Add label

    # Convert the image with bounding boxes to base64 to send to HTML
    _, img_encoded = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(img_encoded).decode('utf-8')

    return render_template('index.html', prediction=label, img_base64=img_base64)  # Pass the image and prediction result to the template

if __name__ == '__main__':
    app.run(debug=True)  # Run the app with debugging enabled
