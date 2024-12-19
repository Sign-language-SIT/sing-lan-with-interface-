import os
import pickle
import mediapipe as mp
import cv2
import numpy as np
import math

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'

def extract_hand_features(img):
    """
    Extract enhanced hand features with multiple representations
    """
    # Convert image to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process image
    results = hands.process(img_rgb)
    
    # Feature vector to store landmarks and additional features
    feature_vector = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract x, y coordinates
            x_coords = [landmark.x for landmark in hand_landmarks.landmark]
            y_coords = [landmark.y for landmark in hand_landmarks.landmark]

            # Normalization parameters
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)

            # Enhanced feature extraction
            for i, landmark in enumerate(hand_landmarks.landmark):
                # Normalized coordinates
                x_norm = (landmark.x - x_min) / (x_max - x_min)
                y_norm = (landmark.y - y_min) / (y_max - y_min)

                # Add normalized coordinates
                feature_vector.append(x_norm)
                feature_vector.append(y_norm)

                # Add angle and distance features if possible
                if i > 0:
                    prev_landmark = hand_landmarks.landmark[i-1]
                    
                    # Angle between consecutive landmarks
                    angle = math.atan2(
                        landmark.y - prev_landmark.y, 
                        landmark.x - prev_landmark.x
                    )
                    
                    # Distance between landmarks
                    distance = math.sqrt(
                        (landmark.x - prev_landmark.x)**2 + 
                        (landmark.y - prev_landmark.y)**2
                    )
                    
                    feature_vector.append(angle)
                    feature_vector.append(distance)

    # If no landmarks detected, return zero vector
    if not feature_vector:
        feature_vector = [0.0] * 84  # 21 landmarks * 4 features (x, y, angle, distance)

    # Truncate or pad to ensure consistent length
    if len(feature_vector) > 84:
        feature_vector = feature_vector[:84]
    elif len(feature_vector) < 84:
        feature_vector.extend([0.0] * (84- len(feature_vector)))

    return feature_vector

# Collect and process data
data = []
labels = []

# Iterate through each class directory
for dir_ in os.listdir(DATA_DIR):
    class_path = os.path.join(DATA_DIR, dir_)
    
    # Ensure it's a directory
    if not os.path.isdir(class_path):
        continue

    # Process each image in the class directory
    for img_path in os.listdir(class_path):
        full_path = os.path.join(class_path, img_path)
        
        # Read image
        img = cv2.imread(full_path)
        
        # Extract features
        features = extract_hand_features(img)
        
        # Add to dataset
        data.append(features)
        labels.append(dir_)

# Save processed dataset
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print(f"Dataset created with {len(data)} samples across {len(set(labels))} classes")