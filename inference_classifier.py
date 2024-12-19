import os
import pickle
import cv2
import mediapipe as mp
import numpy as np
import math
from collections import deque
from time import time

# Load pre-trained model
MODEL_PATH = './models/sign_language_model.pkl'
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("No trained model found. Please run train_classifier.py first.")

# Load model and labels
with open(MODEL_PATH, 'rb') as f:
    model_data = pickle.load(f)
    model = model_data['model']
    labels = model_data.get('labels', [])

# Create label dictionary
labels_dict = {i: chr(65 + i) for i in range(26)}  # A-Z mapping

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_utils.DrawingSpec

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

class SentenceBuilder:
    def __init__(self):
        self.current_word = []
        self.sentence = []
        self.last_prediction = None
        self.last_prediction_time = time()
        self.stable_duration = 1.0  # Time to hold a sign for it to be registered
        self.space_gesture_duration = 1.5  # Time to hold no gesture for space
        self.last_gesture_time = time()
        self.prediction_buffer = deque(maxlen=10)
        self.waiting_for_space = False
        self.space_gesture_count = 0
        
    def update(self, prediction, hand_detected, hand_count=0):
        current_time = time()
        
        # Method 1: Space by absence of hands
        if not hand_detected:
            if self.current_word and (current_time - self.last_gesture_time) >= self.space_gesture_duration:
                self.finish_word()
            self.last_prediction = None
            self.prediction_buffer.clear()
            self.last_gesture_time = current_time
            return
            
        # Method 2: Space by showing two hands briefly
        if hand_count == 2:
            self.space_gesture_count += 1
            if self.space_gesture_count >= 5:  # About 5 frames of showing two hands
                self.finish_word()
                self.space_gesture_count = 0
        else:
            self.space_gesture_count = 0
        
        # Update prediction buffer
        if prediction:
            self.prediction_buffer.append(prediction)
        
        # Check if prediction is stable
        if len(self.prediction_buffer) == self.prediction_buffer.maxlen:
            most_common = max(set(self.prediction_buffer), 
                            key=list(self.prediction_buffer).count)
            buffer_ratio = list(self.prediction_buffer).count(most_common) / len(self.prediction_buffer)
            
            # If prediction is stable and different from last prediction
            if buffer_ratio > 0.8 and most_common != self.last_prediction:
                self.current_word.append(most_common)
                self.last_prediction = most_common
                self.last_prediction_time = current_time
        
        self.last_gesture_time = current_time
    
    def finish_word(self):
        if self.current_word:
            word = ''.join(self.current_word)
            self.sentence.append(word)
            self.current_word = []
    
    def add_space(self):
        self.finish_word()
    
    def get_current_text(self):
        current_word = ''.join(self.current_word)
        return ' '.join(self.sentence + ([current_word] if current_word else []))
    
    def backspace(self):
        if self.current_word:
            self.current_word.pop()
        elif self.sentence:
            last_word = list(self.sentence.pop())
            self.current_word = last_word
    
    def clear(self):
        self.current_word = []
        self.sentence = []
        self.last_prediction = None
        self.prediction_buffer.clear()

def extract_hand_features(hand_landmarks, frame_width, frame_height):
    """
    Extract consistent 84-feature vector for hand landmarks
    """
    x_coords = [landmark.x for landmark in hand_landmarks.landmark]
    y_coords = [landmark.y for landmark in hand_landmarks.landmark]

    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    feature_vector = []

    for i, landmark in enumerate(hand_landmarks.landmark):
        x_norm = (landmark.x - x_min) / (x_max - x_min) if x_max > x_min else 0
        y_norm = (landmark.y - y_min) / (y_max - y_min) if y_max > y_min else 0

        feature_vector.append(x_norm)
        feature_vector.append(y_norm)

        if i > 0:
            prev_landmark = hand_landmarks.landmark[i-1]
            
            angle = math.atan2(
                landmark.y - prev_landmark.y, 
                landmark.x - prev_landmark.x
            )
            
            distance = math.sqrt(
                (landmark.x - prev_landmark.x)**2 + 
                (landmark.y - prev_landmark.y)**2
            )
            
            feature_vector.append(angle)
            feature_vector.append(distance)

    # Ensure exactly 84 features
    if len(feature_vector) > 84:
        feature_vector = feature_vector[:84]
    elif len(feature_vector) < 84:
        feature_vector.extend([0.0] * (84 - len(feature_vector)))

    return feature_vector

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video capture device")
        return

    sentence_builder = SentenceBuilder()
    
    instructions = [
        "Space Controls:",
        "1. Hold no hands up for 1.5s",
        "2. Show two hands briefly",
        "3. Press SPACEBAR",
        "Other Controls:",
        "BACKSPACE - Delete last letter",
        "C - Clear all text",
        "Q - Quit"
    ]

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            try:
                results = hands.process(frame_rgb)
            except Exception as e:
                print(f"Error processing frame: {e}")
                continue

            viz_frame = np.zeros((200, frame.shape[1], 3), dtype=np.uint8)
            
            hand_detected = False
            hand_count = 0
            
            if results.multi_hand_landmarks:
                hand_detected = True
                hand_count = len(results.multi_hand_landmarks)
                
                for hand_landmarks in results.multi_hand_landmarks:
                    try:
                        mp_drawing.draw_landmarks(
                            frame, 
                            hand_landmarks, 
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
                            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                        )

                        features = extract_hand_features(
                            hand_landmarks, frame.shape[1], frame.shape[0]
                        )
                        features_array = np.array(features).reshape(1, -1)
                        prediction = model.predict(features_array)
                        predicted_character = labels_dict[int(prediction[0])]
                        
                        sentence_builder.update(predicted_character, True, hand_count)

                        x_coords = [landmark.x for landmark in hand_landmarks.landmark]
                        y_coords = [landmark.y for landmark in hand_landmarks.landmark]
                        
                        x1 = int(min(x_coords) * frame.shape[1])
                        y1 = int(min(y_coords) * frame.shape[0])
                        x2 = int(max(x_coords) * frame.shape[1])
                        y2 = int(max(y_coords) * frame.shape[0])

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, predicted_character, 
                                    (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 
                                    1, (0, 255, 0), 2)

                    except Exception as e:
                        print(f"Error processing hand landmarks: {e}")
                        continue
            
            else:
                sentence_builder.update(None, False)

            current_text = sentence_builder.get_current_text()
            cv2.putText(viz_frame, "Current Text:", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            words = current_text.split()
            lines = []
            current_line = []
            
            for word in words:
                if len(' '.join(current_line + [word])) * 15 < frame.shape[1] - 20:
                    current_line.append(word)
                else:
                    lines.append(' '.join(current_line))
                    current_line = [word]
            
            if current_line:
                lines.append(' '.join(current_line))
            
            for i, line in enumerate(lines):
                cv2.putText(viz_frame, line, (10, 70 + i * 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            for i, instruction in enumerate(instructions):
                cv2.putText(frame, instruction, (10, 30 + i * 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            combined_frame = np.vstack([frame, viz_frame])
            cv2.imshow('Sign Language Detection', combined_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                sentence_builder.clear()
            elif key == ord(' '):
                sentence_builder.add_space()
            elif key == 8:
                sentence_builder.backspace()

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()

if __name__ == "__main__":
    main()