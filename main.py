import cv2
from ultralytics import YOLO
import pytesseract
import pyttsx3
import speech_recognition as sr
import threading
import time

# Initialize
engine = pyttsx3.init()
model = YOLO("yolov8n.pt")
recognizer = sr.Recognizer()

# Globals
last_spoken = ""
last_time = 0
current_objects = []
command = ""

def speak(text):
    global last_spoken, last_time
    if text != last_spoken or time.time() - last_time > 3:
        engine.say(text)
        engine.runAndWait()
        last_spoken = text
        last_time = time.time()

# 🎙️ Voice command listener (runs in background)
def listen_commands():
    global command
    with sr.Microphone() as source:
        print("Listening for commands...")
        while True:
            try:
                audio = recognizer.listen(source)
                text = recognizer.recognize_google(audio).lower()
                print("Command:", text)
                command = text
            except:
                pass

# Start voice thread
threading.Thread(target=listen_commands, daemon=True).start()

# Distance estimation
def estimate_distance(box_width):
    if box_width == 0:
        return 0
    return round(1000 / box_width, 2)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    detected = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            label = model.names[cls]

            detected.append(label)

            width = x2 - x1
            distance = estimate_distance(width)

            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            text_display = f"{label} | {distance}m"
            cv2.putText(frame, text_display, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0), 2)

    current_objects = list(set(detected))

    # 🎯 HANDLE COMMANDS
    if "what is in front" in command:
        if current_objects:
            speak("I see " + ", ".join(current_objects))
        else:
            speak("I see nothing")
        command = ""

    elif "read text" in command:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray)

        if text.strip() != "":
            speak(text)
        else:
            speak("No text found")
        command = ""

    elif "stop" in command:
        speak("Stopping system")
        break

    cv2.imshow("AI Smart Vision (Voice Enabled)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()