import cv2
import time
from datetime import datetime
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  

cap = cv2.VideoCapture(0)

detection = False
detection_stopped_time = None
timer_started = False
SECONDS_TO_RECORD_AFTER_DETECTION = 5

frame_size = (int(cap.get(3)), int(cap.get(4)))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = None

target_classes = ["person", "cat", "car"]  

print("[INFO] System running. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, stream=True)
    objects_detected = False

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if label in target_classes:
                objects_detected = True

    if objects_detected:
        if not detection:
            detection = True
            timer_started = False
            current_time = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
            out = cv2.VideoWriter(f"{current_time}.mp4", fourcc, 20, frame_size)
            print("[INFO] Started Recording!")

    elif detection:
        if timer_started:
            if time.time() - detection_stopped_time >= SECONDS_TO_RECORD_AFTER_DETECTION:
                detection = False
                timer_started = False
                if out:
                    out.release()
                print("[INFO] Stopped Recording.")
        else:
            timer_started = True
            detection_stopped_time = time.time()

    if detection and out:
        out.write(frame)

    cv2.imshow("YOLOv8 Object Detection", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
if out:
    out.release()
cv2.destroyAllWindows()
