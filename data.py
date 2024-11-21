import cv2
import os

def capture_images(name, num_images=30):
    # Create a directory to save images
    if not os.path.exists(f'dataset/{name}'):
        os.makedirs(f'dataset/{name}')

    cap = cv2.VideoCapture(0)
    count = 0

    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            count += 1
            face = gray[y:y+h, x:x+w]
            cv2.imwrite(f'dataset/{name}/{count}.jpg', face)
            cv2.imshow('Face', face)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        elif count >= num_images:
            break

    cap.release()
    cv2.destroyAllWindows()

# Capture images for a person named 'User'
capture_images('User')
