"""
Smart Intersection - Real-Time Traffic Density Analysis and
Emergency Vehicle Prioritization System

Using a custom-trained YOLOv8n model and the ByteTrack tracker, the system:
  * counts vehicles per lane (inbound / outbound),
  * detects ambulances, fire trucks and police cars and raises a priority alarm,
  * reports lane density with a 5-second time lock to keep the status stable.

Usage:
    python smart_intersection.py                                   # every .mp4 in videos/
    python smart_intersection.py --videos videos/test_police.mp4   # specific video(s)
    python smart_intersection.py --source 0                        # webcam

Keys:  N = next video,  Q = quit
"""

import argparse
import glob
import os
import time

import cv2
from ultralytics import YOLO

# ==========================================
# SETTINGS
# ==========================================
FRAME_W, FRAME_H = 1280, 720
MID_X = FRAME_W // 2          # Left half: INBOUND, right half: OUTBOUND
LINE_Y = 400                  # Virtual counting line
CATCH_BAND = 70               # +/-70 px around the line = 140 px capture zone
RECOUNT_LOCK_SEC = 4.0        # Prevents the same track ID from being counted twice
ALARM_SEC = 5.0               # How long the priority alarm stays on screen
DENSITY_LOCK_SEC = 5.0        # Minimum time a density status stays unchanged
DENSE_ON, DENSE_OFF = 3, 1    # >=3 vehicles: HEAVY, <=1 vehicle: NORMAL (hysteresis)

# Confidence thresholds calibrated for compressed video / CCTV footage
DEFAULT_POLICE_CONF = 0.55
DEFAULT_EMERGENCY_CONF = 0.45

# Keywords matched against model class names
AMB_KEYS = ["ambul", "112", "saglik"]
FIRE_KEYS = ["fire", "itfaiye", "tfaiye"]
POLICE_KEYS = ["polis", "police", "emniyet", "cop"]
HEAVY_KEYS = ["kamyon", "bus", "otobus", "truck", "buyukarac", "belediye"]
IGNORE_KEYS = ["insan", "person"]   # Pedestrians are not counted as vehicles

GREEN, RED, YELLOW = (0, 255, 0), (0, 0, 255), (0, 255, 255)
ORANGE, BLUE, WHITE = (0, 140, 255), (255, 0, 0), (255, 255, 255)


def parse_args():
    p = argparse.ArgumentParser(description="Smart Intersection ITS Control Center")
    p.add_argument("--model", default="models/best.pt", help="YOLOv8 weights file")
    p.add_argument("--videos", nargs="*", help="Video files to play in order")
    p.add_argument("--source", help="Single source (e.g. 0 for webcam or an RTSP URL)")
    p.add_argument("--police-conf", type=float, default=DEFAULT_POLICE_CONF)
    p.add_argument("--emergency-conf", type=float, default=DEFAULT_EMERGENCY_CONF)
    p.add_argument("--det-conf", type=float, default=0.20, help="General detection threshold")
    return p.parse_args()


def classify(cls_name, conf, police_conf, emergency_conf):
    """Maps a model class name to a system category. Returns None for pedestrians."""
    name = cls_name.lower()
    if any(k in name for k in IGNORE_KEYS):
        return None
    if any(k in name for k in AMB_KEYS) and conf >= emergency_conf:
        return "ambulance"
    if any(k in name for k in FIRE_KEYS) and conf >= emergency_conf:
        return "fire_truck"
    if any(k in name for k in POLICE_KEYS) and conf >= police_conf:
        return "police"
    if any(k in name for k in HEAVY_KEYS):
        return "truck_bus"
    return "car"


STYLE = {
    "ambulance": (RED, "AMBULANCE"),
    "fire_truck": (ORANGE, "FIRE TRUCK"),
    "police": (BLUE, "POLICE"),
    "truck_bus": (YELLOW, "Heavy Vehicle"),
    "car": (GREEN, "Car"),
}


class LaneDensity:
    """Lane density status with hysteresis and a time lock to prevent flickering."""

    def __init__(self):
        self.text, self.color, self.locked_until = "NORMAL FLOW", GREEN, 0.0

    def update(self, n_vehicles, now):
        if now < self.locked_until:
            return
        if self.text == "NORMAL FLOW" and n_vehicles >= DENSE_ON:
            self.text, self.color = "HEAVY TRAFFIC", RED
            self.locked_until = now + DENSITY_LOCK_SEC
        elif self.text == "HEAVY TRAFFIC" and n_vehicles <= DENSE_OFF:
            self.text, self.color = "NORMAL FLOW", GREEN
            self.locked_until = now + DENSITY_LOCK_SEC


def draw_panel(frame, x, title, c, density):
    # Panels start below the alarm banner (0-40 px) so the total line is never hidden
    y0 = 48
    cv2.rectangle(frame, (x, y0), (x + 320, y0 + 190), (15, 15, 15), -1)
    cv2.rectangle(frame, (x, y0), (x + 320, y0 + 190), YELLOW, 1)
    rows = [
        (f"{title} TOTAL: {c['total']}", GREEN, 0.70, 2),
        (f"Status       : {density.text}", density.color, 0.50, 2),
        (f"Car          : {c['car']}", WHITE, 0.55, 1),
        (f"Truck/Bus    : {c['truck_bus']}", WHITE, 0.55, 1),
        (f"AMBULANCE    : {c['ambulance']}", RED, 0.55, 2),
        (f"FIRE TRUCK   : {c['fire_truck']}", ORANGE, 0.55, 2),
        (f"POLICE       : {c['police']}", (255, 100, 100), 0.55, 2),
    ]
    for i, (text, color, scale, thick) in enumerate(rows):
        cv2.putText(frame, text, (x + 10, y0 + 25 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)


def main():
    args = parse_args()

    if not os.path.exists(args.model):
        raise SystemExit(f"ERROR: '{args.model}' not found!")
    model = YOLO(args.model)

    if args.source is not None:
        sources = [int(args.source) if args.source.isdigit() else args.source]
    else:
        sources = args.videos or sorted(glob.glob("videos/*.mp4"))
    if not sources:
        raise SystemExit("No videos found. Add .mp4 files to videos/ or use --videos.")

    # Counters are cumulative across videos
    empty = {"total": 0, "car": 0, "truck_bus": 0, "ambulance": 0, "fire_truck": 0, "police": 0}
    counts = {"inbound": dict(empty), "outbound": dict(empty)}
    density = {"inbound": LaneDensity(), "outbound": LaneDensity()}
    last_counted = {}
    alarm_until, alarm_type = 0.0, ""

    print("N: next video | Q: quit")

    for idx, src in enumerate(sources, 1):
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            print(f"WARNING: could not open '{src}', skipping.")
            continue
        print(f"[Video {idx}/{len(sources)}] {src}")

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.resize(frame, (FRAME_W, FRAME_H))
            now = time.time()
            line_color = YELLOW
            live = {"inbound": 0, "outbound": 0}

            res = model.track(frame, persist=True, tracker="bytetrack.yaml",
                              conf=args.det_conf, verbose=False)[0]

            if len(res.boxes) > 0:
                boxes = res.boxes.xyxy.cpu().numpy().astype(int)
                clss = res.boxes.cls.cpu().numpy().astype(int)
                confs = res.boxes.conf.cpu().numpy()
                ids = (res.boxes.id.cpu().numpy().astype(int)
                       if res.boxes.id is not None else [0] * len(boxes))

                for (x1, y1, x2, y2), tid, ci, conf in zip(boxes, ids, clss, confs):
                    cat = classify(res.names[ci], conf, args.police_conf, args.emergency_conf)
                    if cat is None:
                        continue
                    color, label = STYLE[cat]
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    lane = "inbound" if cx < MID_X else "outbound"
                    live[lane] += 1

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.circle(frame, (cx, cy), 5, RED, -1)
                    tag = f"ID:{tid} {label}" if tid else label
                    cv2.putText(frame, f"{tag} ({int(conf * 100)}%)", (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)

                    # Wide capture zone + per-ID re-count lock
                    in_band = abs(cy - LINE_Y) < CATCH_BAND
                    if tid and in_band and now - last_counted.get(tid, -1e9) > RECOUNT_LOCK_SEC:
                        last_counted[tid] = now
                        line_color = GREEN
                        counts[lane]["total"] += 1
                        counts[lane][cat] += 1
                        if cat in ("ambulance", "fire_truck", "police"):
                            alarm_type, alarm_until = label, now + ALARM_SEC

            for lane in ("inbound", "outbound"):
                density[lane].update(live[lane], now)

            cv2.line(frame, (0, LINE_Y), (FRAME_W, LINE_Y), line_color, 3)
            cv2.line(frame, (MID_X, 0), (MID_X, FRAME_H), YELLOW, 2)
            draw_panel(frame, 10, "INBOUND", counts["inbound"], density["inbound"])
            draw_panel(frame, MID_X + 10, "OUTBOUND", counts["outbound"], density["outbound"])
            cv2.putText(frame, f"Playing: {os.path.basename(str(src))} [Video {idx}/{len(sources)}]",
                        (10, FRAME_H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

            if now < alarm_until:
                cv2.rectangle(frame, (0, 0), (FRAME_W, 40), RED, -1)
                cv2.putText(frame,
                            f"EMERGENCY VEHICLE ({alarm_type}) DETECTED - "
                            "PRIORITIZING TRAFFIC LIGHTS!",
                            (30, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)

            cv2.imshow("Smart Intersection - ITS Control Center", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                cap.release()
                cv2.destroyAllWindows()
                return
            if key == ord("n"):
                break

        cap.release()

    cv2.destroyAllWindows()
    print("All videos finished.")


if __name__ == "__main__":
    main()
