from ultralytics import YOLO
import cv2

# Load model (pre-trained)
model = YOLO("yolov8n.pt")

# Start camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # Run detection
    results = model(frame)

    # Show output
    annotated_frame = results[0].plot()
    cv2.imshow("YOLO Detection", annotated_frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()