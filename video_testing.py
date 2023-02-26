import cv2
import numpy as np
from keras.models import load_model

# Load the trained model
model = load_model('proto_model.h5')

# Initialize the camera
cap = cv2.VideoCapture(0)

# Set the frame size
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1240)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1280)

# Define a function for preprocessing the frame
def preprocess_frame(frame):
    # Preprocess the frame (e.g. resize, convert to grayscale, etc.)
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(gray_image, (64, 64))
    normalized_image = resized_image / 255.0
    return normalized_image

# Process video frames in real time
while True:
    # Capture a frame from the camera
    ret, frame = cap.read()

    # Preprocess the frame
    preprocessed_frame = preprocess_frame(frame)

    # Make a prediction using the trained model
    prediction = model.predict(preprocessed_frame[np.newaxis, :, :, np.newaxis])
    prediction_string = str(prediction[0])
    cv2.putText(frame, prediction_string, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Display the frame with the predicted value
    cv2.imshow('frame', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
