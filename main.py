from ultralytics import YOLO
import cv2

print("Loading YOLO model...")

model = YOLO("yolo11n.pt")

print("YOLO model loaded successfully!")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Camera could not be opened.")
    exit()

print("Camera: WORKING")
print("Object detection started!")
print("Press Q to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read camera frame.")
        break

    results = model(frame, verbose=False)

    annotated_frame = results[0].plot()

    cv2.imshow("AI Object Detector", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()