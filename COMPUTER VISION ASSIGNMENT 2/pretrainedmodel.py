import tensorflow_hub as hub
import tensorflow as tf
import cv2
import os
import numpy as np

# Function to load the pre-trained CNN model for emotion classification from TensorFlow Hub
# Set TensorFlow logging level to suppress informational messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom


def load_emotion_model(model_url):
    model = tf.keras.Sequential([
        hub.KerasLayer(model_url, input_shape=[], dtype=tf.string)
    ])
    return model

# Function to detect faces and return bounding box coordinates


def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=5)
    return faces, gray

# Function to classify emotion using the loaded CNN model


def classify_emotion(face_roi, model):
    # Resize to match model input shape
    resized_roi = cv2.resize(face_roi, (224, 224))
    resized_roi = np.expand_dims(resized_roi, axis=0)
    resized_roi = resized_roi / 255.0  # Normalize
    predictions = model.predict(resized_roi)
    emotion_label = emotions[np.argmax(predictions)]
    return emotion_label


# Map emotion labels to human-readable emotions
emotions = {0: 'Angry', 1: 'Disgust', 2: 'Fear',
            3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

# Load the pre-trained model for emotion classification from TensorFlow Hub
model_url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"  # Example URL
emotion_model = load_emotion_model(model_url)

# Load the uploaded image
image_path = r'C:\\Users\\hp\Pictures\\mood\\WhatsApp Image 2024-06-28 at 16.12.34_cca91be9.jpg'
image = cv2.imread(image_path)

# Detect faces in the image
faces, gray = detect_faces(image)

# Process each detected face
for (x, y, w, h) in faces:
    face_roi = gray[y:y+h, x:x+w]  # Region of interest (face) in grayscale
    emotion = classify_emotion(face_roi, emotion_model)
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0),
                  2)  # Draw rectangle around the face
    cv2.putText(image, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (36, 255, 12), 2)  # Display emotion label

# Display the image with detected faces and emotions
cv2.imshow('Detected Emotions', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
