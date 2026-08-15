# 🛡️ SafeZone CV: Real-Time Proximity & Safety Analytics

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://safezone-cv-eshwar.streamlit.app)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange.svg)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green.svg)](https://opencv.org/)

An automated computer vision and spatial analytics dashboard built with **YOLOv8**, **OpenCV**, and **Streamlit**. SafeZone CV monitors video feeds in real-time, calculates inter-person Euclidean distances, flags workplace/public safety violations, and logs dynamic density metrics.

🔗 **Live Demo:** [safezone-cv-eshwar.streamlit.app](https://safezone-cv-eshwar.streamlit.app)

---

## 📌 Key Features

* **Real-Time Entity Detection:** Employs the lightweight YOLOv8 Nano model for low-latency personnel detection from video streams.
* **Spatial Proximity Engine:** Computes pairwise Euclidean distances across detected bounding-box centroids to flag non-compliant proximity events.
* **Dynamic Visual Alerts:** Renders visual bounding boxes (Green for compliant, Red for proximity violations) along with connective violation vectors in real time.
* **Interactive Control Dashboard:** Allows dynamic adjustments of the detection confidence threshold and pixel proximity boundary sliders via Streamlit.
* **Live Incident Logging:** Visualizes frame-by-frame entity counts and violation spikes using integrated Pandas data pipelines and line charts.

---

## 🛠️ Architecture & Pipeline
[Video Stream / Upload]
│
▼
[YOLOv8 Feature Extraction & Detection]
│
▼
[Centroid Computation & Spatial Distance Matrix]
│
├───► Distance < Threshold ──► Flag Violation & Draw Red Vector
└───► Distance ≥ Threshold ──► Safe Status (Green Bounding Box)
│
▼
[Streamlit UI + Real-Time Metrics & Trend Analytics]


---

## 🚀 Quickstart & Local Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/eshwar2206/SafeZone-CV.git](https://github.com/eshwar2206/SafeZone-CV.git)
cd SafeZone-CV
2. Create and Activate Virtual Environment
Bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
3. Install Dependencies
Bash
pip install -r requirements.txt
4. Run the Application
To run the Streamlit Web Dashboard:

Bash
streamlit run app.py
To run standalone OpenCV video processing:

Bash
python detector.py
📦 Tech Stack
Language: Python 3.11

Deep Learning Framework: YOLOv8 (ultralytics-opencv-headless)

Computer Vision: OpenCV

Data Processing & Analytics: NumPy, Pandas

Deployment & UI: Streamlit Cloud

👤 Author
GitHub: @eshwar2206


---

### How to Add it to GitHub

Run these commands in your VS Code terminal:

```bash
git add README.md
git commit -m "Add professional README documentation"
git push origin main
