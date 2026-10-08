# First version (baseline): direction- and type-based vehicle counting with the
# pre-trained COCO YOLOv8n model.
# Run: python baseline/vehicle_counting_coco.py  (yolov8n.pt is downloaded automatically)

import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
video_path = "videos/test_video.mp4"  # put your own test video here
cap = cv2.VideoCapture(video_path)

# --- COUNTING VARIABLES ---
offset = 15
counted_ids = []

# Counters per direction and vehicle type
outbound = {"total": 0, "car": 0, "truck": 0, "bus": 0}
inbound = {"total": 0, "car": 0, "truck": 0, "bus": 0}

# COCO class IDs -> names
class_names = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Video finished or could not be read.")
        break

    results = model.track(frame, persist=True, classes=[2, 3, 5, 7])
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

                    # Left lane: outbound
                    if cx < mid_x:
                        outbound["total"] += 1
                        if vehicle_type in outbound:
                            outbound[vehicle_type] += 1
                    # Right lane: inbound
                    else:
                        inbound["total"] += 1
                        if vehicle_type in inbound:
                            inbound[vehicle_type] += 1

    # --- ON-SCREEN REPORT ---
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

    cv2.imshow("Vehicle Counting - Baseline", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
