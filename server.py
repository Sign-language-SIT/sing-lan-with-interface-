import asyncio
import websockets
import json
import cv2
import numpy as np
import pickle
import mediapipe as mp
import math
from collections import deque
import base64
import os
from time import time
import logging


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Sentence Builder class
class SentenceBuilder:
    def __init__(self):
        self.current_word = []
        self.sentence = []
        self.last_prediction = None
        self.last_prediction_time = time()
        self.stable_duration = 1.0
        self.space_gesture_duration = 1.5
        self.last_gesture_time = time()
        self.prediction_buffer = deque(maxlen=10)
        self.waiting_for_space = False
        self.space_gesture_count = 0
        
    def update(self, prediction, hand_detected, hand_count=0):
        current_time = time()
        
        if not hand_detected:
            if self.current_word and (current_time - self.last_gesture_time) >= self.space_gesture_duration:
                self.finish_word()
            self.last_prediction = None
            self.prediction_buffer.clear()
            self.last_gesture_time = current_time
            return
            
        if hand_count == 2:
            self.space_gesture_count += 1
            if self.space_gesture_count >= 5:
                self.finish_word()
                self.space_gesture_count = 0
        else:
            self.space_gesture_count = 0
        
        if prediction:
            self.prediction_buffer.append(prediction)
        
        if len(self.prediction_buffer) == self.prediction_buffer.maxlen:
            most_common = max(set(self.prediction_buffer), 
                            key=list(self.prediction_buffer).count)
            buffer_ratio = list(self.prediction_buffer).count(most_common) / len(self.prediction_buffer)
            
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

# Feature extraction function
def extract_hand_features(hand_landmarks, frame_width, frame_height):
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

# Load model
MODEL_PATH = './models/sign_language_model.pkl'
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("No trained model found. Please run train_classifier.py first.")

with open(MODEL_PATH, 'rb') as f:
    model_data = pickle.load(f)
    model = model_data['model']
    labels = model_data.get('labels', [])

# Create label dictionary
labels_dict = {i: chr(65 + i) for i in range(26)}

class SignLanguageServer:
    async def process_frame(self, frame_data, websocket):
            try:
                # Log frame processing
                logger.debug("Processing new frame")
                
                # Decode base64 image
                jpg_as_np = np.frombuffer(base64.b64decode(frame_data), dtype=np.uint8)
                frame = cv2.imdecode(jpg_as_np, cv2.IMREAD_COLOR)
                
                if frame is None:
                    logger.error("Failed to decode frame")
                    return
                    
                logger.debug(f"Frame shape: {frame.shape}")
                
                # Process frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame_rgb)
                
                if results.multi_hand_landmarks:
                    logger.debug(f"Detected {len(results.multi_hand_landmarks)} hands")
                    
                    for hand_landmarks in results.multi_hand_landmarks:
                        features = extract_hand_features(
                            hand_landmarks, frame.shape[1], frame.shape[0]
                        )
                        features_array = np.array(features).reshape(1, -1)
                        
                        logger.debug(f"Feature vector shape: {features_array.shape}")
                        
                        prediction = model.predict(features_array)
                        predicted_character = labels_dict[int(prediction[0])]
                        
                        logger.debug(f"Predicted character: {predicted_character}")

                        self.sentence_builders[websocket].update(
                            predicted_character, True, len(results.multi_hand_landmarks)
                        )
                else:
                    logger.debug("No hands detected")
                    self.sentence_builders[websocket].update(None, False)
                    
                # Send current text to client
                current_text = self.sentence_builders[websocket].get_current_text()
                await websocket.send(json.dumps({
                    'text': current_text
                }))
                
            except Exception as e:
                logger.error(f"Error processing frame: {str(e)}", exc_info=True)

    def __init__(self):
        self.clients = set()
        self.sentence_builders = {}
        
    async def register(self, websocket):
        self.clients.add(websocket)
        self.sentence_builders[websocket] = SentenceBuilder()
        
    async def unregister(self, websocket):
        self.clients.remove(websocket)
        del self.sentence_builders[websocket]

    async def process_frame(self, frame_data, websocket):
        try:
            # Decode base64 image
            jpg_as_np = np.frombuffer(base64.b64decode(frame_data), dtype=np.uint8)
            frame = cv2.imdecode(jpg_as_np, cv2.IMREAD_COLOR)
            
            # Process frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            
            hand_detected = False
            hand_count = 0
            
            if results.multi_hand_landmarks:
                hand_detected = True
                hand_count = len(results.multi_hand_landmarks)
                
                for hand_landmarks in results.multi_hand_landmarks:
                    features = extract_hand_features(
                        hand_landmarks, frame.shape[1], frame.shape[0]
                    )
                    features_array = np.array(features).reshape(1, -1)
                    prediction = model.predict(features_array)
                    predicted_character = labels_dict[int(prediction[0])]

                    self.sentence_builders[websocket].update(
                        predicted_character, True, hand_count
                    )
            else:
                self.sentence_builders[websocket].update(None, False)
                
            # Send current text to client
            current_text = self.sentence_builders[websocket].get_current_text()
            await websocket.send(json.dumps({
                'text': current_text
            }))
        except Exception as e:
            print(f"Error processing frame: {e}")   

    async def handle_command(self, command, websocket):
        try:
            if command == 'clear':
                self.sentence_builders[websocket].clear()
            elif command == 'space':
                self.sentence_builders[websocket].add_space()
            elif command == 'backspace':
                self.sentence_builders[websocket].backspace()
                
            # Send updated text
            current_text = self.sentence_builders[websocket].get_current_text()
            await websocket.send(json.dumps({
                'text': current_text
            }))
        except Exception as e:
            print(f"Error handling command: {e}")

    async def handler(self, websocket):
        await self.register(websocket)
        try:
            async for message in websocket:
                data = json.loads(message)
                
                if 'frame' in data:
                    await self.process_frame(data['frame'], websocket)
                elif 'command' in data:
                    await self.handle_command(data['command'], websocket)
                else:
                    print("Unknown message format received.")
        except websockets.exceptions.ConnectionClosedError:
            print("Client disconnected.")
        except Exception as e:
            print(f"Error in WebSocket handler: {e}")
        finally:
            await self.unregister(websocket)

async def main():
    server = SignLanguageServer()
    async with websockets.serve(server.handler, "localhost", 8765):
        print("Server running on ws://localhost:8765")
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())
