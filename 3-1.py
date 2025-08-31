import cv2
import time
from datetime import datetime
import argparse
import os

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")

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

    if len(faces) + len(bodies) > 0:
        if not detection:
            detection = True
            timer_started = False
            current_time = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
            out = cv2.VideoWriter(f"{current_time}.mp4", fourcc, 20, frame_size)
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
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)

    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
if out:
    out.release()
cv2.destroyAllWindows()

ap = argparse.ArgumentParser()
ap.add_argument("-ext", "--extension", required=False, default="jpg")
ap.add_argument("-o", "--output", required=False, default="output.mp4")
args = vars(ap.parse_args())

dir_path = "."
ext = args["extension"]
output = args["output"]

images = [f for f in os.listdir(dir_path) if f.endswith(ext)]
if images:
    images.sort()  
    first_image_path = os.path.join(dir_path, images[0])
    frame = cv2.imread(first_image_path)
    height, width, channels = frame.shape

    out = cv2.VideoWriter(output, fourcc, 5.0, (width, height))

    for image in images:
        image_path = os.path.join(dir_path, image)
        frame = cv2.imread(image_path)
        out.write(frame)

    out.release()
    print(f"Video saved as {output}")
