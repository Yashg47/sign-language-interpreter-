import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define input shape (assuming grayscale images)
IMG_SIZE = (64, 64)  
INPUT_SHAPE = (64, 64, 1)  # 1 for grayscale images
NUM_CLASSES = 26  # A-Z (26 classes)

# Define dataset path
DATASET_PATH = "dataset/data"  # Ensure this is the correct path

# Data augmentation & normalization
datagen = ImageDataGenerator(
    rescale=1.0/255.0,  # Normalize pixel values
    validation_split=0.2  # 80-20 train-validation split
)

# Load Training Data
train_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=32,
    class_mode="categorical",
    subset="training"
)

# Load Validation Data
val_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=32,
    class_mode="categorical",
    subset="validation"
)

print("Dataset Loaded Successfully!")

# Define CNN Model
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=INPUT_SHAPE),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(NUM_CLASSES, activation='softmax')  # 26 output classes
])

# Compile the Model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Print model summary
model.summary()

# Train the CNN Model
EPOCHS = 20  # You can adjust this based on performance

history = model.fit(
    train_data,
    epochs=EPOCHS,
    validation_data=val_data
)

# Save the Model
model.save("sign_language_cnn.h5")
    


print("Model training completed and saved as 'sign_language_cnn.h5'.")

