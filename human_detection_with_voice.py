import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox
from gtts import gTTS
import os

# Function to generate and play a voice alert
def generate_voice_alert(message):
    tts = gTTS(text=message, lang='en')  # Generate text-to-speech
    tts.save("alert.mp3")  # Save it as an audio file
    os.system("start alert.mp3")  # Play the audio file (Windows: 'start', macOS: 'afplay', Linux: 'mpg123')

# Open the camera
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Detect humans
    bbox, label, conf = cv.detect_common_objects(frame, model='yolov3-tiny')

    # Check if a human is detected
    if 'person' in label:
        print("Human detected!")
        generate_voice_alert("Human detected!")  # Generate a voice alert

    # Draw bounding boxes around detected objects
    output_frame = draw_bbox(frame, bbox, label, conf)

    # Show the video with bounding boxes
    cv2.imshow("Human Detection with Voice Alert", output_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
