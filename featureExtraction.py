import math
from sklearn.preprocessing import StandardScaler

# Function to calculate distance between two landmarks
def calculate_distance(landmark1, landmark2):
    return math.sqrt((landmark2.x - landmark1.x) ** 2 + (landmark2.y - landmark1.y) ** 2)

# Function to calculate angle between three landmarks
def calculate_angle(landmark1, landmark2, landmark3):
    # Using the dot product formula to calculate angle
    vec1 = [landmark1.x - landmark2.x, landmark1.y - landmark2.y]
    vec2 = [landmark3.x - landmark2.x, landmark3.y - landmark2.y]

    dot_product = vec1[0] * vec2[0] + vec1[1] * vec2[1]
    mag1 = math.sqrt(vec1[0]**2 + vec1[1]**2)
    mag2 = math.sqrt(vec2[0]**2 + vec2[1]**2)

    cos_theta = dot_product / (mag1 * mag2)
    angle = math.acos(cos_theta)  # Angle in radians

    return angle

# Function to extract features from hand landmarks
def extract_features(landmarks):
    thumb_tip = landmarks.landmark[4]  # Thumb tip
    index_tip = landmarks.landmark[8]  # Index tip
    middle_tip = landmarks.landmark[12]  # Middle finger tip
    ring_tip = landmarks.landmark[16]  # Ring finger tip
    pinky_tip = landmarks.landmark[20]  # Pinky tip
    wrist = landmarks.landmark[0]  # Wrist (useful for orientation)

    # Distances between key points
    distance_thumb_pinky = calculate_distance(thumb_tip, pinky_tip)
    distance_thumb_index = calculate_distance(thumb_tip, index_tip)
    distance_index_pinky = calculate_distance(index_tip, pinky_tip)
    distance_wrist_thumb = calculate_distance(wrist, thumb_tip)
    distance_wrist_pinky = calculate_distance(wrist, pinky_tip)

    # Angles between key points
    angle_thumb_index_middle = calculate_angle(thumb_tip, index_tip, middle_tip)
    angle_index_middle_ring = calculate_angle(index_tip, middle_tip, ring_tip)
    angle_middle_ring_pinky = calculate_angle(middle_tip, ring_tip, pinky_tip)
    angle_thumb_wrist = calculate_angle(thumb_tip, wrist, index_tip)  # Wrist-to-thumb angle

    # Ratios of distances (can help detect hand orientation and size)
    thumb_index_ratio = distance_thumb_index / distance_thumb_pinky if distance_thumb_pinky != 0 else 0
    pinky_index_ratio = distance_index_pinky / distance_thumb_pinky if distance_thumb_pinky != 0 else 0

    # Return all the features
    features = [
        distance_thumb_pinky,
        distance_thumb_index,
        distance_index_pinky,
        distance_wrist_thumb,
        distance_wrist_pinky,
        angle_thumb_index_middle,
        angle_index_middle_ring,
        angle_middle_ring_pinky,
        angle_thumb_wrist,
        thumb_index_ratio,
        pinky_index_ratio
    ]
    
    # Normalize the features (using StandardScaler)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform([features])[0]  # Normalize the feature vector
    return scaled_features

