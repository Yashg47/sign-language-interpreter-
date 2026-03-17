import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from PIL import Image  # Using PIL instead of OpenCV
from glob import glob  # For efficient file searching

# Define constants
IMG_SIZE = 224  
data_dir = "dataset/data"  # Change to your actual image folder path
csv_file = "dataset.csv"  # CSV must contain 'image_name' and 'label' columns

# Ensure CSV file exists
if not os.path.exists(csv_file):
    raise FileNotFoundError(f"CSV file not found: {csv_file}")

# Ensure image directory exists
if not os.path.exists(data_dir):
    raise FileNotFoundError(f"Image folder not found: {data_dir}")

# Load dataset
df = pd.read_csv(csv_file)

# Drop rows with missing values
df.dropna(subset=["image_name", "label"], inplace=True)

# Track missing/corrupt images
missing_images = []
corrupt_images = []

# Initialize lists
images = []
labels = []

# Check available images
existing_files = set(os.path.basename(f) for f in glob(os.path.join(data_dir, "*")))

# Process images
for index, row in df.iterrows():
    img_filename = row["image_name"]
    img_path = os.path.join(data_dir, img_filename)

    if img_filename not in existing_files:
        missing_images.append(img_filename)
        continue  # Skip this file

    try:
        # Read and preprocess image using PIL
        img = Image.open(img_path).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img = np.array(img) / 255.0  # Normalize

        images.append(img)
        labels.append(row["label"])

    except Exception as e:
        corrupt_images.append(img_filename)
        print(f"Error reading {img_filename}: {e}")

# Convert lists to NumPy arrays
images = np.array(images, dtype=np.float32)
labels = np.array(labels)

# Print missing/corrupt image summary
print(f"Total Missing Images: {len(missing_images)}")
print(f"Total Corrupt Images: {len(corrupt_images)}")

# Encode labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)
labels = to_categorical(labels)  # Convert to one-hot encoding

# Split dataset (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Print dataset shapes
print(f"Train Shape: {X_train.shape}, Test Shape: {X_test.shape}")

# Save processed data (optional)
np.save("X_train.npy", X_train)
np.save("X_test.npy", X_test)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)
np.save("label_classes.npy", label_encoder.classes_)  # Save label mapping

print("Dataset processing complete and saved successfully!")

