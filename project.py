import cv2
from ultralytics import YOLO

# Load your trained YOLO model (replace 'your_model.pt' with the path to your .pt file)
model = YOLO(r"C:\Users\saiki\Downloads\best.pt")  # e.g., 'yolo_traffic.pt'

# Open the webcam (use 0 for the default camera, or replace with a video path)
cap = cv2.VideoCapture(0)  # You can replace 0 with a video file path if using a video

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Run YOLO inference on the frame
    results = model(frame)

    # Loop through the detected objects
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()  # Confidence score
            cls = int(box.cls[0].item())  # Class index
            
            # Get the class name
            class_name = model.names[cls]
            
            # Draw the bounding box and label on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f'{class_name}: {conf:.2f}'
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('YOLO Traffic Signal Detection', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()