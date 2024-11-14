import cv2
import mediapipe as mp
import pandas as pd
import numpy as np

# Initialize MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize VideoCapture
cap = cv2.VideoCapture(0)

# Create a background subtractor
backSub = cv2.createBackgroundSubtractorMOG2()

# Ask user for the alphabet to capture
alphabet = input("Enter the alphabet you are capturing: ").strip().upper()

# Function to capture hand landmarks
def capture_hand_data():
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

    # Initialize sample counter
    sample_count = 0
    max_samples = 400  # Set maximum samples limit

    with hands as hand_model:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            # Apply background subtraction
            fg_mask = backSub.apply(image)

            # Convert the BGR image to RGB and process with MediaPipe
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hand_model.process(image_rgb)

            # Combine the mask with the original image to isolate hands
            combined = cv2.bitwise_and(image, image, mask=fg_mask)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Collecting the landmarks (21 points with (x, y, z) coordinates)
                    hand_data = []
                    for landmark in hand_landmarks.landmark:
                        hand_data.extend([landmark.x, landmark.y, landmark.z])

                    # Draw the landmarks on the combined image
                    mp_drawing.draw_landmarks(combined, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            cv2.imshow('Hand Tracking', combined)

            key = cv2.waitKey(5) & 0xFF

            if key == ord('s'):  # If 's' is pressed, save the data to CSV
                if results.multi_hand_landmarks and sample_count < max_samples:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Collect the hand landmarks and append the alphabet label
                        hand_data = []
                        for landmark in hand_landmarks.landmark:
                            hand_data.extend([landmark.x, landmark.y, landmark.z])
                        hand_data.append(alphabet)  # Add the label to the data

                        # Save data to CSV
                        save_to_csv(hand_data)
                        sample_count += 1  # Increment the sample count
                        print(f"Data saved for alphabet '{alphabet}' ({sample_count}/{max_samples}).")

                        # Break after saving to avoid saving multiple times for the same press
                        break

                    if sample_count >= max_samples:
                        print(f"Maximum samples reached for alphabet '{alphabet}'. No more data will be saved.")

            elif key == ord('q'):  # If 'q' is pressed, quit the application
                print("Exiting...")
                break

    cap.release()
    cv2.destroyAllWindows()

# Function to save data into a CSV file
def save_to_csv(data):
    df = pd.DataFrame([data])
    df.to_csv('./hand_data.csv', mode='a', header=False, index=False)

# Run the capture function
capture_hand_data()