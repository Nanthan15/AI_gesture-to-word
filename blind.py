import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import pyttsx3

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Define the gesture-to-word mapping
gesture_to_word = {
    "thumbs_up": "yes",
    "thumbs_down": "no",
    "okay": "okay",
    "peace": "peace"
}

# Function to convert text to speech
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Initialize the HandDetector class with the given parameters
detector = HandDetector(staticMode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5)

# Function to detect specific gestures
def detect_gesture(hand):
    lmList = hand["lmList"]
    
    # Thumb up: Thumb is up and index finger is down
    if lmList[4][2] < lmList[3][2] and lmList[8][2] > lmList[6][2]:
        return "thumbs_up"
    
    # Thumb down: Thumb is down and index finger is up
    if lmList[4][2] > lmList[3][2] and lmList[8][2] < lmList[6][2]:
        return "thumbs_down"
    
    # Okay: Thumb and index finger form a circle (distance between tip of thumb and index finger is small)
    thumb_tip = lmList[4][:2]  # Get (x, y) coordinates of thumb tip
    index_tip = lmList[8][:2]  # Get (x, y) coordinates of index tip
    if detector.findDistance(thumb_tip, index_tip)[0] < 30:
        return "okay"
    
    # Peace: Index and middle fingers are up, other fingers are down
    if lmList[8][2] < lmList[6][2] and lmList[12][2] < lmList[10][2] and lmList[16][2] > lmList[14][2] and lmList[20][2] > lmList[18][2]:
        return "peace"
    
    return None

# Continuously get frames from the webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while True:
    success, img = cap.read()
    if not success:
        break
    hands, img = detector.findHands(img, draw=True, flipType=True)

    if hands:
        for hand in hands:
            gesture = detect_gesture(hand)
            if gesture in gesture_to_word:
                word = gesture_to_word[gesture]
                print(f"Gesture detected: {word}")  # Debug print
                speak(word)
                cv2.putText(img, word, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
