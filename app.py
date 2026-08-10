import cv2
import math
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO

# Streamlit Page Layout Setup
st.set_page_config(
    page_title="SafeZone CV - Analytics Dashboard",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ SafeZone CV: Real-Time Proximity & Safety Analytics")
st.markdown("Upload video footage to monitor personnel density and spatial distance violations in real-time.")

# Sidebar Configuration
st.sidebar.header("Control Panel")
uploaded_file = st.sidebar.file_uploader("Upload Video File", type=["mp4", "avi", "mov"])
proximity_threshold = st.sidebar.slider("Proximity Threshold (Pixels)", min_value=50, max_value=300, value=150, step=10)
confidence_threshold = st.sidebar.slider("YOLO Detection Confidence", min_value=0.1, max_value=0.9, value=0.4, step=0.05)

# Helper Functions
def get_box_center(box):
    x1, y1, x2, y2 = box
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))

def calculate_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

model = load_model()

if uploaded_file is not None:
    # Save uploaded file to temporary storage for OpenCV reader
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)

    # UI Dashboard Columns
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("Live Processing Feed")
        frame_window = st.image([])

    with col2:
        st.subheader("Real-Time Metrics")
        kpi_detected = st.empty()
        kpi_violations = st.empty()
        
    # Real-time Chart Data Collector
    analytics_data = []
    chart_placeholder = st.empty()

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Run YOLO Inference
        results = model(frame, conf=confidence_threshold, verbose=False)[0]
        
        person_boxes = []
        centers = []

        # Filter Person Class (cls == 0)
        for box in results.boxes:
            if int(box.cls[0]) == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                person_boxes.append((x1, y1, x2, y2))
                centers.append(get_box_center((x1, y1, x2, y2)))

        violating_indexes = set()
        num_persons = len(centers)

        # Distance Evaluation Matrix
        for i in range(num_persons):
            for j in range(i + 1, num_persons):
                dist = calculate_distance(centers[i], centers[j])
                if dist < proximity_threshold:
                    violating_indexes.add(i)
                    violating_indexes.add(j)
                    cv2.line(frame, centers[i], centers[j], (0, 0, 255), 2)

        # Bounding Box Overlay
        for i, (x1, y1, x2, y2) in enumerate(person_boxes):
            color = (0, 0, 255) if i in violating_indexes else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Convert OpenCV BGR image to RGB for Streamlit display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_window.image(frame_rgb, channels="RGB", use_container_width=True)

        # Update KPI Cards
        kpi_detected.metric(label="Total Persons Detected", value=num_persons)
        kpi_violations.metric(label="Active Proximity Violations", value=len(violating_indexes))

        # Append to analytics log
        analytics_data.append({"Frame": frame_count, "People": num_persons, "Violations": len(violating_indexes)})
        
        # Periodically update the real-time line graph
        if frame_count % 5 == 0:
            df_analytics = pd.DataFrame(analytics_data)
            chart_placeholder.line_chart(df_analytics.set_index("Frame"))

    cap.release()
    st.success("Video processing complete!")

else:
    st.info("Please upload a video file using the sidebar to begin processing.")