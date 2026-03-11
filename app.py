import streamlit as st
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import tempfile
import os

# Set page config
st.set_page_config(page_title="Traffic Monitoring System", layout="wide")

st.title("🚦 Smart Traffic Monitoring System")
st.markdown("""
This application uses **YOLOv8** for vehicle detection and **DeepSORT** for real-time tracking.
It monitors speed, detects wrong-way driving, and logs traffic events.
""")

# Cache the model
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# Sidebar configuration
st.sidebar.header("Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.3)
roi_start_ratio = st.sidebar.slider("ROI Start Height (Ratio)", 0.0, 1.0, 0.3)
speed_limit = st.sidebar.number_input("Speed Alert Threshold", value=7)

# File uploader
uploaded_file = st.file_uploader("Upload a traffic video", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Save uploaded file to temp
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Resize for processing speed if too large
    target_width, target_height = 640, 480
    
    # Initialize Tracker
    tracker = DeepSort(max_age=30, n_init=3, max_cosine_distance=0.4)
    
    # Containers for display
    col1, col2 = st.columns([2, 1])
    video_placeholder = col1.empty()
    event_placeholder = col2.empty()
    
    # State management
    track_history = {}
    vehicle_records = {} # Latest details for each unique track
    event_log = []
    frame_count = 0
    
    stop_button = st.button("Stop Processing")
    
    progress_bar = st.progress(0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    st.info("Processing video... Results will appear below.")

    # Create placeholders for live dashboards
    st.subheader("Live Vehicle Records")
    live_record_placeholder = st.empty()

    while cap.isOpened() and not stop_button:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        frame = cv2.resize(frame, (target_width, target_height))
        h, w, _ = frame.shape

        # Draw ROI Boundary and Divider
        divider_x = int(target_width * 0.5)
        roi_y_start = int(h * roi_start_ratio)
        cv2.line(frame, (divider_x, 0), (divider_x, h), (255, 255, 0), 1) # Vertical divider
        cv2.line(frame, (0, roi_y_start), (w, roi_y_start), (255, 0, 255), 2) # ROI boundary

        roi = frame[roi_y_start:h, 0:w]

        # Detection
        results = model(
            roi,
            classes=[2, 3, 5, 7], # car, motorcycle, bus, truck
            conf=conf_threshold,
            imgsz=640,
            verbose=False
        )

        detections = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            cls = int(box.cls[0].cpu().numpy())
            # Convert to [x, y, w, h] for DeepSORT
            detections.append(([x1, y1, x2-x1, y2-y1], conf, cls))

        # Tracking
        tracks = tracker.update_tracks(detections, frame=roi)

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            l, t, r, b = track.to_ltrb()
            
            cls = track.get_det_class()
            label_name = model.names[int(cls)] if cls is not None else "Unknown"

            center_x = int((l + r) / 2)
            center_y = int((t + b) / 2)
            
            speed_flag = 0
            wrong_flag = 0
            pixel_speed = 0.0
            direction = "Unknown"

            # Trajectory
            if track_id not in track_history:
                track_history[track_id] = []
            track_history[track_id].append((center_x, center_y))
            if len(track_history[track_id]) > 20:
                track_history[track_id] = track_history[track_id][-20:]

            points = track_history[track_id]

            # Speed & Direction Logic
            if len(points) >= 5:
                dx = points[-1][0] - points[-5][0]
                dy = points[-1][1] - points[-5][1]
                pixel_speed = np.sqrt(dx**2 + dy**2) / 5

                # Speed alert
                if pixel_speed > speed_limit:
                    speed_flag = 1
                
                # Direction logic
                if center_x < divider_x:
                    if dy < -1: # Moving up
                        direction = "Up (Left Lane)"
                        if dy < -3: wrong_flag = 1
                    elif dy > 1:
                        direction = "Down (Left Lane)"
                    else:
                        direction = "Stopped"
                else:
                    if dy > 1: # Moving down
                        direction = "Down (Right Lane)"
                        if dy > 3: wrong_flag = 1
                    elif dy < -1:
                        direction = "Up (Right Lane)"
                    else:
                        direction = "Stopped"

            # Update vehicle record
            vehicle_records[track_id] = {
                "Vehicle ID": track_id,
                "Type": label_name,
                "Speed": round(pixel_speed, 2),
                "Direction": direction,
                "Last Seen Frame": frame_count,
                "Violation": "None" if not (speed_flag or wrong_flag) else 
                             "Speeding" if speed_flag and not wrong_flag else 
                             "Wrong Way" if wrong_flag and not speed_flag else "Both"
            }

            # Visualization on ROI
            color = (0, 255, 0)
            if speed_flag or wrong_flag:
                color = (0, 0, 255)
            
            cv2.rectangle(roi, (int(l), int(t)), (int(r), int(b)), color, 2)
            cv2.putText(roi, f"{label_name} ID:{track_id}", (int(l), int(t)-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            if wrong_flag:
                cv2.putText(roi, "WRONG WAY", (int(l), int(b)+15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

            # Draw trajectory
            for i in range(1, len(points)):
                cv2.line(roi, points[i-1], points[i], (255, 0, 0), 1)

            if speed_flag or wrong_flag:
                event_log.append({
                    "Frame": frame_count,
                    "ID": track_id,
                    "Type": label_name,
                    "Event": "Speeding" if speed_flag and not wrong_flag else 
                             "Wrong Way" if wrong_flag and not speed_flag else "Both"
                })

        # Update frame back
        frame[roi_y_start:h, 0:w] = roi
        
        # Convert BGR to RGB for Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
        
        # Update event log display
        if event_log:
            df_log = pd.DataFrame(event_log).tail(5)
            event_placeholder.dataframe(df_log, use_container_width=True)

        # Update Live Vehicle Dashboard
        if vehicle_records:
            df_records = pd.DataFrame(list(vehicle_records.values()))
            live_record_placeholder.dataframe(df_records, use_container_width=True)
        
        if total_frames > 0:
            progress_bar.progress(frame_count / total_frames)

    cap.release()
    tfile.close()
    if os.path.exists(video_path):
        os.unlink(video_path)

    st.success("Processing Complete!")
    
    if vehicle_records:
        st.subheader("Full Vehicle Data Report")
        full_df = pd.DataFrame(list(vehicle_records.values()))
        st.dataframe(full_df)
        
        # Save to CSV in project directory
        full_df.to_csv("vehicle_records.csv", index=False)
        st.info("Full record saved as `vehicle_records.csv` in project folder.")

        csv = full_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Vehicle Records CSV", csv, "vehicle_records.csv", "text/csv")
    
    if event_log:
        st.subheader("Event Summary")
        event_df = pd.DataFrame(event_log)
        st.dataframe(event_df)
        
        csv_events = event_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Event Log CSV", csv_events, "traffic_events.csv", "text/csv")
else:
    st.info("Please upload a video file to begin.")
