import cv2
import time
from datetime import datetime
import argparse
import os

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")
profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")
upperbody_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_upperbody.xml")
lowerbody_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_lowerbody.xml")
fullbody_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")
cat_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalcatface.xml")
plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")

video = cv2.VideoCapture(0)

while True:
    check, frame = video.read()
    if frame is not None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.1, 10)
        eyes = eye_cascade.detectMultiScale(gray, 1.1, 10)
        smiles = smile_cascade.detectMultiScale(gray, 1.8, 20)
        profiles = profile_cascade.detectMultiScale(gray, 1.1, 5)
        uppers = upperbody_cascade.detectMultiScale(gray, 1.1, 5)
        lowers = lowerbody_cascade.detectMultiScale(gray, 1.1, 5)
        fulls = fullbody_cascade.detectMultiScale(gray, 1.1, 5)
        cats = cat_cascade.detectMultiScale(gray, 1.1, 5)
        plates = plate_cascade.detectMultiScale(gray, 1.1, 5)

        detections = [
            ("Face", faces, (0, 255, 0)),
            ("Eye", eyes, (255, 0, 0)),
            ("Smile", smiles, (0, 255, 255)),
            ("Profile", profiles, (255, 100, 100)),
            ("UpperBody", uppers, (200, 0, 255)),
            ("LowerBody", lowers, (0, 200, 255)),
            ("FullBody", fulls, (100, 100, 255)),
            ("Cat", cats, (255, 150, 50)),
            ("Plate", plates, (0, 255, 100))
        ]

        detected = False
        for label, objects, color in detections:
            for (x, y, w, h) in objects:
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                detected = True

        if detected:
            exact_time = datetime.now().strftime('%Y-%b-%d-%H-%M-%S-%f')
            cv2.imwrite("detected_" + exact_time + ".jpg", frame)

        cv2.imshow("All Haarcascades Detection", frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            ap = argparse.ArgumentParser()
            ap.add_argument("-ext", "--extension", required=False, default='jpg')
            ap.add_argument("-o", "--output", required=False, default='output.mp4')
            args = vars(ap.parse_args())

            dir_path = '.'
            ext = args['extension']
            output = args['output']

            images = [f for f in os.listdir(dir_path) if f.endswith(ext)]
            images.sort()  

            if images:
                image_path = os.path.join(dir_path, images[0])
                frame = cv2.imread(image_path)
                height, width, channels = frame.shape

                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output, fourcc, 5.0, (width, height))

                for image in images:
                    image_path = os.path.join(dir_path, image)
                    frame = cv2.imread(image_path)
                    out.write(frame)

                out.release()
                print(f"[INFO] Video saved as: {output}")
            break

video.release()
cv2.destroyAllWindows()
