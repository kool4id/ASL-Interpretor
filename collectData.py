import cv2
import mediapipe as mp
import numpy as np
import pickle



# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

gesture_data = []  # List to store gesture data
current_gesture = ""  # To track current gesture being recorded

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            # Extract landmarks as features
            landmarks_list = np.array([[lm.x, lm.y] for lm in landmarks.landmark]).flatten()

            # Display the landmarks
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

            # Show the gesture name being recorded
            cv2.putText(frame, f"Recording {current_gesture}...", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Show video feed
    cv2.imshow("Hand Gesture Recording", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('1'):  # '1' key for A
        current_gesture = "A"
        gesture_data.append({"label": "A", "features": landmarks_list})
        print("Recording 'A' gesture...")

    elif key == ord('2'):  # '2' key for B
        current_gesture = "B"
        gesture_data.append({"label": "B", "features": landmarks_list})
        print("Recording 'B' gesture...")

    elif key == ord('3'):  # '3' key for C
        current_gesture = "C"
        gesture_data.append({"label": "C", "features": landmarks_list})
        print("Recording 'C' gesture...")

    elif key == ord('q'):  # 'q' key to quit and save data
        break

# Save the recorded data to a file
with open('gesture_data.pkl', 'wb') as f:
    pickle.dump(gesture_data, f)

print("Gesture data saved to 'gesture_data.pkl'")

cap.release()
cv2.destroyAllWindows()