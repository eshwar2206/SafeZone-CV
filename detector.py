import cv2
import math
import numpy as np
from ultralytics import YOLO

# 1. Load pre-trained YOLOv8 Nano model (downloads automatically on first run)
model = YOLO('yolov8n.pt')

# 2. Open input video stream
video_path = "test_video.mp4"
cap = cv2.VideoCapture(video_path)

# Distance threshold in pixels (adjust based on camera angle/resolution)
PROXIMITY_THRESHOLD = 150 

def get_box_center(box):
    """Calculates the center point (x, y) of a bounding box."""
    x1, y1, x2, y2 = box
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    return (center_x, center_y)

def calculate_distance(p1, p2):
    """Calculates Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

print("Starting video processing... Press 'q' on the video window to stop.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # End of video

    # Run YOLOv8 detection on the frame (Class 0 corresponds to 'person' in COCO dataset)
    results = model(frame, verbose=False)[0]
    
    person_boxes = []
    centers = []

    # Filter detections to keep only 'person' (cls == 0)
    for box in results.boxes:
        cls = int(box.cls[0])
        if cls == 0:  
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            person_boxes.append((x1, y1, x2, y2))
            centers.append(get_box_center((x1, y1, x2, y2)))

    # Track indexes of people who violate the distance threshold
    violating_indexes = set()

    # Compare every pair of detected persons
    num_persons = len(centers)
    for i in range(num_persons):
        for j in range(i + 1, num_persons):
            dist = calculate_distance(centers[i], centers[j])
            if dist < PROXIMITY_THRESHOLD:
                violating_indexes.add(i)
                violating_indexes.add(j)
                # Draw red connection line between close individuals
                cv2.line(frame, centers[i], centers[j], (0, 0, 255), 2)

    # Draw bounding boxes and status labels
    for i, (x1, y1, x2, y2) in enumerate(person_boxes):
        if i in violating_indexes:
            color = (0, 0, 255)  # Red for proximity warning
            label = "WARNING: Too Close"
        else:
            color = (0, 255, 0)  # Green for safe distance
            label = "SAFE"

        # Draw bounding rectangle and text label
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display real-time violation metrics overlay
    stats_text = f"Total Detected: {num_persons} | Violations: {len(violating_indexes)}"
    cv2.putText(frame, stats_text, (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    # Render frame in window
    cv2.imshow("SafeZone CV - Real-Time Proximity Analytics", frame)

    # Exit stream on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()