import cv2
import time
from datetime import datetime
import argparse
import os

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")
profileface_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")
upperbody_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_upperbody.xml")
lowerbody_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_lowerbody.xml")
catface_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalcatface.xml")
plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")

detection = False
detection_stopped_time = None
timer_started = False
SECONDS_TO_RECORD_AFTER_DETECTION = 5

frame_size = (int(cap.get(3)), int(cap.get(4)))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = None  

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    bodies = body_cascade.detectMultiScale(gray, 1.3, 5)
    eyes = eye_cascade.detectMultiScale(gray, 1.1, 10)
    smiles = smile_cascade.detectMultiScale(gray, 1.7, 22)
    profiles = profileface_cascade.detectMultiScale(gray, 1.1, 4)
    uppers = upperbody_cascade.detectMultiScale(gray, 1.1, 5)
    lowers = lowerbody_cascade.detectMultiScale(gray, 1.1, 5)
    cats = catface_cascade.detectMultiScale(gray, 1.1, 5)
    plates = plate_cascade.detectMultiScale(gray, 1.1, 5)

    if any([len(faces), len(bodies), len(eyes), len(smiles), len(profiles), len(uppers), len(lowers), len(cats), len(plates)]):
        if not detection:
            detection = True
            timer_started = False
            current_time = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
            out = cv2.VideoWriter(f"{current_time}.mp4", fourcc, 5, frame_size)
            print("Started Recording!")

    elif detection:
        if timer_started:
            if time.time() - detection_stopped_time >= SECONDS_TO_RECORD_AFTER_DETECTION:
                detection = False
                timer_started = False
                if out:
                    out.release()
                print("Stop Recording!")
        else:
            timer_started = True
            detection_stopped_time = time.time()

    if detection and out:
        out.write(frame)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    for (x, y, w, h) in bodies:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    for (x, y, w, h) in eyes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 1)
    for (x, y, w, h) in smiles:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 1)
    for (x, y, w, h) in profiles:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    for (x, y, w, h) in uppers:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (200, 0, 255), 2)
    for (x, y, w, h) in lowers:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 255), 2)
    for (x, y, w, h) in cats:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 100, 100), 2)
    for (x, y, w, h) in plates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 255, 100), 2)

    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
if out:
    out.release()
cv2.destroyAllWindows()
