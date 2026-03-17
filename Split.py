from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Assume X contains images, and y contains labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# Convert labels to categorical (One-Hot Encoding) since we have multiple classes
y_train = to_categorical(y_train, num_classes=26)
y_val = to_categorical(y_val, num_classes=26)
y_test = to_categorical(y_test, num_classes=26)
