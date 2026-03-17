import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from PIL import Image  # Use Pillow for image processing

# Define constants
IMG_SIZE = 224  
data_dir = "dataset\data"  # Path to dataset root folder

# Get class names (folder names)
class_names = sorted(os.listdir(data_dir))  # Extract class labels from folder names
print("Detected Classes:", class_names)

# Initialize lists
images = []
labels = []

# Process images
for label, class_name in enumerate(class_names):
    class_folder = os.path.join(data_dir, class_name)
    
    if not os.path.isdir(class_folder):  # Ignore files, only process folders
        continue

    # Iterate over all images in the class folder
    for img_name in os.listdir(class_folder):
        img_path = os.path.join(class_folder, img_name)

        try:
            # Open and preprocess image
            img = Image.open(img_path).convert("RGB")
            img = img.resize((IMG_SIZE, IMG_SIZE))
            img = np.array(img) / 255.0  # Normalize pixel values

            # Append to dataset
            images.append(img)
            labels.append(class_name)  # Store folder name as label

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

# Convert lists to NumPy arrays
images = np.array(images, dtype=np.float32)
labels = np.array(labels)

# # # Encode labels
# # # label_encoder = LabelEncoder()
# # # labels = label_encoder.fit_transform(labels)  # Convert class names to numbers
# # # labels = to_categorical(labels)  # Convert to
# for class_name in os.listdir(dataset/data):
#     class_path = os.path.join(dataset/data, class_name)

#     if os.path.isdir(class_path):  # ✅ Ensure it's a folder
#         for filename in os.listdir(class_path):
#             file_path = os.path.join(class_path, filename)

#             if os.path.isfile(file_path):  # ✅ Ensure it's a file
#                 with open(file_path, "rb") as f:
#                     data = f.read()
