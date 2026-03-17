from tensorflow.keras.models import load_model
import cv2
import numpy as np

# Load trained model
model = load_model("sign_language_cnn.h5")

# Load and preprocess an image
img = cv2.imread(r"C:\Users\Vikhyat\Desktop\web-site\dataset\data\Y\42.jpg", cv2.IMREAD_GRAYSCALE)

img = cv2.resize(img, (64, 64)).reshape(1, 64, 64, 1) / 255.0

# Predict the sign
prediction = model.predict(img)
predicted_label = chr(65 + np.argmax(prediction))  # Convert to A-Z

print("Predicted Sign:", predicted_label)
