import cv2
import mediapipe as mp
from gtts import gTTS
import os

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Function to generate and play a voice alert
def generate_voice_alert(message):
    tts = gTTS(text=message, lang='en')  # Convert text to speech
    tts.save("alert.mp3")  # Save as an MP3 file
    os.system("start alert.mp3")  # Play the MP3 file (use 'afplay' on macOS, 'mpg123' on Linux)

# Open the camera feed
cap = cv2.VideoCapture(0)

while True:
    # Capture frame from the camera
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Convert the frame to RGB (MediaPipe works in RGB, not BGR)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame for hand landmarks
    results = hands.process(rgb_frame)

    # If any hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks on the detected hands
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get hand positions (e.g., wrist, thumb)
            wrist_x = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x
            wrist_y = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y
            thumb_tip_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x
            thumb_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y

            # Gesture: Hunger (rubbing stomach)
            if wrist_x < 0.5 and wrist_y > 0.4:  # Detects hand near stomach
                print("Hunger gesture detected!")
                generate_voice_alert("The person is hungry and wants food!")
            
            # Gesture: Running (both hands moving quickly)
            if thumb_tip_x < wrist_x and thumb_tip_y < wrist_y:  # Detects running motion
                print("Running gesture detected!")
                generate_voice_alert("The person is running!")

            # Gesture: Sleeping (head down, body relaxed)
            if wrist_x > 0.4 and wrist_y < 0.3:  # Detects sleeping pose (head/arms down)
                print("Sleeping gesture detected!")
                generate_voice_alert("The person is sleeping!")

    # Show the live camera feed with hand landmarks
    cv2.imshow("Gesture Recognition", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
