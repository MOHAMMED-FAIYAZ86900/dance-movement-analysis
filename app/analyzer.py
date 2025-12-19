import cv2
import os
import uuid
import numpy as np

import mediapipe as mp
from ultralytics import YOLO


# ==============================
# CONFIGURATION (CPU OPTIMIZED)
# ==============================

DETECTION_INTERVAL = 10      # Run YOLO every N frames
BOX_EXPANSION = 0.20         # Expand person crop by 20%
SMOOTH_ALPHA = 0.7           # Temporal smoothing factor

# ==============================
# INITIALIZE MODELS
# ==============================

# YOLOv8 (CPU-safe)
yolo = YOLO("yolov8n.pt")

# MediaPipe Pose (CPU-tuned)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,          # ⚡ faster + stable on CPU
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# ==============================
# HELPERS
# ==============================

def smooth(prev, curr):
    if prev is None:
        return curr
    return SMOOTH_ALPHA * prev + (1 - SMOOTH_ALPHA) * curr


def expand_box(x, y, w, h, width, height):
    pad_w = int(w * BOX_EXPANSION)
    pad_h = int(h * BOX_EXPANSION)

    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(width, x + w + pad_w)
    y2 = min(height, y + h + pad_h)

    return x1, y1, x2, y2


# ==============================
# MAIN FUNCTION
# ==============================

def analyze_video(input_video_path: str) -> str:

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    os.makedirs("outputs", exist_ok=True)
    output_path = f"outputs/skeleton_{uuid.uuid4().hex}.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    trackers = {}
    prev_landmarks = {}
    next_person_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # ==============================
        # YOLO DETECTION (LOW FREQUENCY)
        # ==============================
        if frame_count % DETECTION_INTERVAL == 0:
            trackers.clear()
            prev_landmarks.clear()
            next_person_id = 0

            results = yolo(frame, conf=0.4, iou=0.5, verbose=False)

            for r in results:
                for box in r.boxes:
                    if int(box.cls[0]) == 0:  # person
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        w, h = x2 - x1, y2 - y1

                        tracker = cv2.TrackerKCF_create()
                        tracker.init(frame, (x1, y1, w, h))

                        trackers[next_person_id] = tracker
                        prev_landmarks[next_person_id] = None
                        next_person_id += 1

        # ==============================
        # TRACK + POSE (EVERY FRAME)
        # ==============================
        for pid, tracker in trackers.items():
            success, box = tracker.update(frame)
            if not success:
                continue

            x, y, w, h = map(int, box)
            x1, y1, x2, y2 = expand_box(x, y, w, h, width, height)

            person_crop = frame[y1:y2, x1:x2]
            if person_crop.size == 0:
                continue

            rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)

            if not result.pose_landmarks:
                continue

            landmarks = []
            for lm in result.pose_landmarks.landmark:
                px = int(lm.x * (x2 - x1)) + x1
                py = int(lm.y * (y2 - y1)) + y1
                landmarks.append([px, py])

            landmarks = np.array(landmarks)
            landmarks = smooth(prev_landmarks[pid], landmarks)
            prev_landmarks[pid] = landmarks

            # ==============================
            # DRAW SKELETON
            # ==============================
            for a, b in mp_pose.POSE_CONNECTIONS:
                cv2.line(
                    frame,
                    tuple(landmarks[a]),
                    tuple(landmarks[b]),
                    (0, 255, 0),
                    2
                )

            for (x, y) in landmarks:
                cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)

        out.write(frame)

    cap.release()
    out.release()

    return output_path
