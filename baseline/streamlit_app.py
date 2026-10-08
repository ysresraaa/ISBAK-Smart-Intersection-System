# Streamlit web dashboard for the first (baseline) version.
# Run: streamlit run baseline/streamlit_app.py

import cv2
import streamlit as st
from ultralytics import YOLO

# 1. Page settings
st.set_page_config(page_title="Traffic Analysis", page_icon="🚦", layout="wide")

st.title("🚦 Smart Intersection Vehicle Classification and Density Analysis")
st.markdown("This dashboard uses a YOLOv8 model to analyze live traffic flow, "
            "directions and vehicle types.")


# Cache the model so it is not reloaded on every page refresh
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")


model = load_model()
video_path = "videos/test_video.mp4"  # put your own test video here

# 2. Sidebar
st.sidebar.header("🎛️ Control Panel")
st.sidebar.markdown("Use the buttons below to start or stop the analysis.")
start_button = st.sidebar.button("▶️ Start Analysis", use_container_width=True)
stop_button = st.sidebar.button("⏹️ Stop", use_container_width=True)

# 3. Placeholder for the video frames
stframe = st.empty()

if start_button:
    cap = cv2.VideoCapture(video_path)

    offset = 15
    counted_ids = []
    outbound = {"total": 0, "car": 0, "truck": 0, "bus": 0}
    inbound = {"total": 0, "car": 0, "truck": 0, "bus": 0}
    class_names = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

    while cap.isOpened() and not stop_button:
        ret, frame = cap.read()
        if not ret:
            st.warning("Video finished or could not be read.")
            break

        # YOLOv8 + ByteTrack tracking
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", classes=[2, 3, 5, 7])
        annotated_frame = results[0].plot()

        height, width, _ = annotated_frame.shape
        line_y = int(height / 2) + 30
        cv2.line(annotated_frame, (0, line_y), (width, line_y), (255, 0, 0), 3)

        mid_x = int(width / 2)
        cv2.line(annotated_frame, (mid_x, 0), (mid_x, height), (0, 255, 255), 3)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()

            for box, track_id, cls_id in zip(boxes, track_ids, class_ids):
                x1, y1, x2, y2 = box
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                cv2.circle(annotated_frame, (cx, cy), 5, (0, 0, 255), -1)

                if (line_y - offset) < cy < (line_y + offset):
                    if track_id not in counted_ids:
                        counted_ids.append(track_id)
                        vehicle_type = class_names.get(cls_id, "car")

                        if cx < mid_x:
                            outbound["total"] += 1
                            if vehicle_type in outbound:
                                outbound[vehicle_type] += 1
                        else:
                            inbound["total"] += 1
                            if vehicle_type in inbound:
                                inbound[vehicle_type] += 1

        # Draw the counters
        cv2.putText(annotated_frame, f"Outbound Total: {outbound['total']}", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        cv2.putText(annotated_frame, f"Car: {outbound['car']}", (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Truck: {outbound['truck']}", (30, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Bus: {outbound['bus']}", (30, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(annotated_frame, f"Inbound Total: {inbound['total']}", (mid_x + 30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        cv2.putText(annotated_frame, f"Car: {inbound['car']}", (mid_x + 30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(annotated_frame, f"Truck: {inbound['truck']}", (mid_x + 30, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(annotated_frame, f"Bus: {inbound['bus']}", (mid_x + 30, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 4. OpenCV uses BGR, Streamlit expects RGB
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        stframe.image(annotated_frame, channels="RGB", use_container_width=True)

    cap.release()
