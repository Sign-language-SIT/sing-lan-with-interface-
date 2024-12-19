import os
import cv2

# Ensure data directory exists
DATA_DIR = './data'
os.makedirs(DATA_DIR, exist_ok=True)

# Configuration
NUMBER_OF_CLASSES = 26  # A-Z
DATASET_SIZE_PER_CLASS = 100  # Number of images per class

def collect_data():
    # Initialize webcam
    cap = cv2.VideoCapture(0)

    # Collect data for each class (A-Z)
    for j in range(NUMBER_OF_CLASSES):
        # Create class-specific directory
        class_dir = os.path.join(DATA_DIR, str(j))
        os.makedirs(class_dir, exist_ok=True)

        print(f'Collecting data for class {chr(65 + j)} (Press "Q" when ready)')

        # Wait for user to be ready
        while True:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)  # Mirror the frame
            
            # Display instruction
            cv2.putText(frame, 
                        f'Preparing to collect images for {chr(65 + j)}. Press "Q" when ready!', 
                        (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, (0, 255, 0), 2)
            
            cv2.imshow('Data Collection', frame)
            
            # Check for 'q' key
            if cv2.waitKey(25) == ord('q'):
                break

        # Collect images
        counter = 0
        while counter < DATASET_SIZE_PER_CLASS:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)  # Mirror the frame
            
            # Display collection progress
            cv2.putText(frame, 
                        f'Collecting {chr(65 + j)}: {counter}/{DATASET_SIZE_PER_CLASS}', 
                        (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, (0, 255, 0), 2)
            
            cv2.imshow('Data Collection', frame)
            cv2.waitKey(25)
            
            # Save image
            cv2.imwrite(os.path.join(class_dir, f'{counter}.jpg'), frame)
            counter += 1

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

def main():
    collect_data()

if __name__ == "__main__":
    main()