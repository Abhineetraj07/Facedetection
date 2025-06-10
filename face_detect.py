import cv2
#pip install  opencv-python

cascade_face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open the webcam
cap = cv2.VideoCapture(0)

# Check if webcam is opened
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, img = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break


    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    f = cascade_face.detectMultiScale(g, 1.3, 4)

    # Draw rectangles around detected faces
    for (x, y, w, h) in f:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 4)

    # Display the frame
    cv2.imshow('Face Detection', img)

    # Exit when 'ESC' is pressed
    if cv2.waitKey(30) & 0xFF == 27:
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
