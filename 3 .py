import cv2, time
from datetime  import datetime
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


video = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    bodies = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) + len(bodies) > 0:
        if detection:
            timer_started = False
        else:
            detection = True
            current_time = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
            out = cv2.VideoWriter(f"{current_time}.mp4", fourcc, 30, frame_size)
            print("Started Recording!")

    elif detection:
        if timer_started:
            if time.time() - detection_stopped_time >= SECONDS_TO_RECORD_AFTER_DETECTION:
                detection = False
                timer_started = False
                out.release()
                print('Stop Recording!')
        else:
            timer_started = True
            detection_stopped_time = time.time()

    if detection:
        out.write(frame)

    for (x, y, width, height) in faces:
        cv2.rectangle(frame, (x,y), (x + width, y + height), (255, 0, 0),3)



    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) == ord('q'):
        break

while True:
    check,frame=video.read()
    if frame is not None:
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=10)
        for x,y,w,h in faces:
            img=cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
            exact_time=datetime.now().strftime('%Y-%b-%d-%H-%S-%f')
            cv2.imwrite("face detected"+str(exact_time)+".jpg",img)


        cv2.imshow("home surv",frame)
        key=cv2.waitKey(1)

        if key==ord('q'):
            ap=argparse.ArgumentParser()
            ap.add_argument("-ext","--extension",required=False,default='jpg')
            ap.add_argument("-o","--output",required=False,default='output.mp4')
            args=vars(ap.parse_args())


            dir_path='.'
            ext=args['extension']
            output=args['output']


            images=[]

            for f in os.listdir(dir_path):
                if f.endswith(ext):
                    images.append(f)



            image_path=os.path.join(dir_path,images[0])
            frame=cv2.imread(image_path)
            height,width,channels=frame.shape


            forcc=cv2.VideoWriter_fourcc(*'mp4v')
            out=cv2.VideoWriter(output,forcc,5.0,(width,height))


            for image in images:
                image_path=os.path.join(dir_path,image)
                frame=cv2.imread(image_path)
                out.write(frame)

            break

out.release()
video.release()
cv2.destroyAllWindows