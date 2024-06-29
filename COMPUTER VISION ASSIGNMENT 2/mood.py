import cv2

# Replace with the path where you saved haarcascade_frontalface_default.xml
cascade_path = r'C:\Users\hp\Desktop\mood\haarcascade_frontalface_default.xml'

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cascade_path)
