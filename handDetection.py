import cv2
import mediapipe as mp
import numpy as np
import joblib

# Load the trained model and scaler
knn = joblib.load('gesture_recognition_model.pkl')  # Load the trained KNN model
scaler = joblib.load('gesture_scaler.pkl')  # Load the scaler

# Initialize MediaPipe hands for hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils  # For drawing hand landmarks on the frame

# Initialize webcam
cap = cv2.VideoCapture(0)

def calculate_distance(landmark1, landmark2):
    """Calculate Euclidean distance between two landmarks"""
    return np.sqrt((landmark2.x - landmark1.x) ** 2 + (landmark2.y - landmark1.y) ** 2)

def extract_features(landmarks):
    """Extract features from hand landmarks (e.g., distances between fingertips)"""
    thumb_tip = landmarks.landmark[4]  # Thumb tip
    pinky_tip = landmarks.landmark[20]  # Pinky tip
    index_tip = landmarks.landmark[8]  # Index tip

    # Calculate distances between fingertips
    distance_thumb_pinky = calculate_distance(thumb_tip, pinky_tip)
    distance_thumb_index = calculate_distance(thumb_tip, index_tip)

    return [distance_thumb_pinky, distance_thumb_index]  # Return the features

while True:
    ret, frame = cap.read()  # Read a frame from the webcam
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Flip the frame horizontally for a mirror effect
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB (required by MediaPipe)
    results = hands.process(rgb_frame)  # Process the frame with MediaPipe Hands

    if results.multi_hand_landmarks:  # If hands are detected
        for landmarks in results.multi_hand_landmarks:
            # Extract features from the landmarks
            features = extract_features(landmarks)

            # Scale the features using the scaler (same scaler used in training)
            features_scaled = scaler.transform([features])

            # Predict the gesture using the trained KNN model
            prediction = knn.predict(features_scaled)
            gesture = prediction[0]  # Get the predicted gesture (e.g., "a", "b", etc.)

            # Display the predicted gesture on the frame
            cv2.putText(frame, f"{gesture}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Draw hand landmarks and connections on the frame
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the frame with the predictions and hand landmarks
    cv2.imshow("Hand Gesture Detection", frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()