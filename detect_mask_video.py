import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the trained model
model = load_model('D:/ComputerVision/MaskDetection/mask_classifier_model.keras')  # Provide the correct path to your saved model

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the webcam
cap = cv2.VideoCapture(0)  # Use '0' for the default webcam

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Camera not found.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture image.")
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Loop through the detected faces
    for (x, y, w, h) in faces:
        # Crop the face from the frame
        face = frame[y:y + h, x:x + w]
        face_resized = cv2.resize(face, (224, 224))  # Resize to match model's input size
        face_array = img_to_array(face_resized) / 255.0  # Normalize
        face_array = np.expand_dims(face_array, axis=0)  # Add batch dimension

        # Predict the class of the face (Mask/No Mask)
        pred = model.predict(face_array)[0][0]
        label = "Mask" if pred < 0.5 else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        # Draw bounding box and label on the frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Display the frame with bounding boxes
    cv2.imshow("Mask Detection", frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the window
cap.release()
cv2.destroyAllWindows()
