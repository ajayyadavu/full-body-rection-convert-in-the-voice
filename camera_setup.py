import cv2

# Open the camera
cap = cv2.VideoCapture(0)  # 0 refers to the default camera

while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break
    
    # Display the frame
    cv2.imshow("Camera Feed", frame)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()


