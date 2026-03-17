import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("sign_language_cnn.h5")
          
# Define image size
IMG_SIZE = (64, 64)

# Start webcam
cap = cv2.VideoCapture(0)  # 0 is the default webcam

while True:
    ret, frame = cap.read()  # Capture frame
    if not ret:
        break  # If no frame is captured, exit loop

    # Convert frame to grayscale (same as training images)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Define region of interest (ROI) - You can adjust this for better results
    roi = cv2.resize(gray, IMG_SIZE)  # Resize to match model input
    roi = roi.reshape(1, 64, 64, 1) / 255.0  # Reshape & normalize

    # Predict the letter
    prediction = model.predict(roi)
    predicted_label = chr(65 + np.argmax(prediction))  # Convert index to letter (A-Z)

    # Display the prediction
    cv2.putText(frame, f"Prediction: {predicted_label}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the webcam frame
    cv2.imshow("Sign Language Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close window
cap.release()
cv2.destroyAllWindows()

