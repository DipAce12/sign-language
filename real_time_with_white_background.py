import cv2
import mediapipe as mp
import numpy as np
import joblib
import tensorflow as tf

# Load the trained CNN model and scaler
try:
    model = tf.keras.models.load_model("./cnn_hand_model.h5")
    scaler = joblib.load('./scaler.pkl')
    labels = np.load('./labels.npy', allow_pickle=True)  # Load the saved labels
except Exception as e:
    print(f"Error loading model/scaler/labels: {e}")
    exit(1)

# Initialize MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize the video capture
cap = cv2.VideoCapture(0)

# Optionally set frame size for the camera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def real_time_recognition():
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

    with hands as hand_model:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Error: Unable to read from the camera.")
                break

            # Create a white background image
            white_background = np.ones_like(image) * 255

            # Convert the BGR image to RGB for MediaPipe
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hand_model.process(image_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    hand_data = []
                    for landmark in hand_landmarks.landmark:
                        hand_data.extend([landmark.x, landmark.y, landmark.z])

                    # Convert hand data to a numpy array and reshape for CNN input
                    hand_data = np.array(hand_data).reshape(1, -1)  # Shape: (1, 63)
                    hand_data_scaled = scaler.transform(hand_data)  # Scale the data

                    # Reshape for model input (1 sample, 63 features, 1 channel)
                    hand_data_reshaped = hand_data_scaled.reshape(1, 63, 1)  # Shape: (1, 63, 1)

                    # Predict using the CNN model
                    prediction = model.predict(hand_data_reshaped)
                    predicted_label_index = np.argmax(prediction)
                    predicted_label = labels[predicted_label_index]

                    # Confidence threshold
                    confidence = np.max(prediction)
                    if confidence >= 0.5:  # Show predictions above 50% confidence
                        print(f"Predicted Sign: {predicted_label} (Confidence: {confidence:.2f})")
                        cv2.putText(white_background, predicted_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    else:
                        print("Prediction confidence too low")

                    # Draw the landmarks on the white background
                    mp_drawing.draw_landmarks(white_background, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Display the image with hand landmarks on the white background
            cv2.imshow('Real-time Hand Tracking', white_background)

            # Press 'ESC' to exit
            if cv2.waitKey(5) & 0xFF == 27:
                break

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()

# Run the real-time recognition
if __name__ == "__main__":
    real_time_recognition()
