import cv2
import mediapipe as mp
import pandas as pd
import numpy as np

# Initialize MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize background subtractor
backSub = cv2.createBackgroundSubtractorMOG2()

cap = cv2.VideoCapture(0)

# Ask user for the alphabet to capture
alphabet = input("Enter the alphabet you are capturing: ").strip().upper()

# Create a function to capture hand landmarks with a white background
def capture_hand_data():
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
    sample_count = 0
    max_samples = 200  # Set maximum samples limit

    with hands as hand_model:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            # Apply background subtraction to enhance hand isolation
            fg_mask = backSub.apply(image)

            # Create a white background image
            white_background = np.ones_like(image) * 255

            # Process with MediaPipe
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hand_model.process(image_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    hand_data = []
                    for landmark in hand_landmarks.landmark:
                        hand_data.extend([landmark.x, landmark.y, landmark.z])

                    # Draw landmarks on the white background
                    mp_drawing.draw_landmarks(white_background, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Display the hand tracking on the white background
            cv2.imshow('Hand Tracking', white_background)

            key = cv2.waitKey(5) & 0xFF

            if key == ord('s'):  # Save data to CSV
                if results.multi_hand_landmarks and sample_count < max_samples:
                    for hand_landmarks in results.multi_hand_landmarks:
                        hand_data = []
                        for landmark in hand_landmarks.landmark:
                            hand_data.extend([landmark.x, landmark.y, landmark.z])
                        hand_data.append(alphabet)

                        save_to_csv(hand_data)
                        sample_count += 1
                        print(f"Data saved for alphabet '{alphabet}' ({sample_count}/{max_samples}).")
                        break

                    if sample_count >= max_samples:
                        print(f"Maximum samples reached for alphabet '{alphabet}'.")

            elif key == ord('q'):
                print("Exiting...")
                break

    cap.release()
    cv2.destroyAllWindows()

# Function to save data into a CSV file
def save_to_csv(data):
    df = pd.DataFrame([data])
    df.to_csv('hand_data.csv', mode='a', header=False, index=False)

# Run the capture function
capture_hand_data()
