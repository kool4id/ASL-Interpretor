import numpy as np
import pickle
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Example gesture data (You can replace this with your actual recorded data from collect_data.py)
gesture_data = [
    {"features": [0.34, 0.45], "label": "a"},
    {"features": [0.31, 0.42], "label": "b"},
    {"features": [0.25, 0.38], "label": "c"},
    {"features": [0.41, 0.39], "label": "d"},
    {"features": [0.35, 0.50], "label": "e"},
    # Add more gestures for other letters as needed...
]

# Extract features (X) and labels (y) from the gesture_data
X = [data["features"] for data in gesture_data]  # List of features
y = [data["label"] for data in gesture_data]  # List of corresponding labels (letters)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the StandardScaler to scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Scale the training data
X_test_scaled = scaler.transform(X_test)  # Scale the testing data using the same scaler

# Initialize and train the K-Nearest Neighbors (KNN) classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_scaled, y_train)

# Test the model on the test set
y_pred = knn.predict(X_test_scaled)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the trained model and scaler to files
joblib.dump(knn, 'gesture_recognition_model.pkl')  # Save the trained KNN model
joblib.dump(scaler, 'gesture_scaler.pkl')  # Save the scaler

print("Model and scaler saved successfully!")