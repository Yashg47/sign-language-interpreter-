import os
import pandas as pd

# Path to the main dataset folder (Modify as needed)
data_dir = "dataset/data"

# List to store image information
data = []

# Loop through each subfolder (category/label)
for category in os.listdir(data_dir):
    category_path = os.path.join(data_dir, category)

    # Ensure it's a directory (skip files)
    if os.path.isdir(category_path):
        # Loop through images in the category folder
        for img_name in os.listdir(category_path):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):  # Only process images
                img_path = os.path.join(category, img_name)  # Store relative path
                data.append([img_path, category])  # Append to list

# Convert to a DataFrame
df = pd.DataFrame(data, columns=["image_name", "label"])

# Save CSV file
csv_filename = "dataset.csv"
df.to_csv(csv_filename, index=False)

print(f"CSV file created successfully: {csv_filename}")


