import cv2
import os
import uuid
import numpy as np

import mediapipe as mp
from ultralytics import YOLO


# -----------------------------
# INITIALIZE MODELS (ONCE)
# -----------------------------

# YOLOv8 for person detection
yolo = YOLO("yolov8n.pt")  # lightweight + fast

# MediaPipe Pose (single-person, high accuracy)
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    smooth_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)


# -----------------------------
# HELPER: SMOOTH LANDMARKS
# -----------------------------

def smooth_landmarks(prev, curr, alpha=0.7):
    """Exponential moving average smoothing"""
    if prev is None:
        return curr
    return alpha * prev + (1 - alpha) * curr


# -----------------------------
# MAIN ANALYSIS FUNCTION
# -----------------------------

def analyze_video(input_video_path: str) -> str:
    """
    Analyze a dance video and return path to skeleton-overlay video
    """

    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        raise RuntimeError("Could not open video")

    # Video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Output file
    output_name = f"skeleton_{uuid.uuid4().hex}.mp4"
    output_path = os.path.join("outputs", output_name)

    os.makedirs("outputs", exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Store previous landmarks per person index
    prev_landmarks = {}

    # -----------------------------
    # PROCESS FRAMES
    # -----------------------------

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize for consistency
        frame = cv2.resize(frame, (width, height))

        # Detect people using YOLO
        results = yolo(frame, verbose=False)

        person_boxes = []

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls == 0:  # class 0 = person
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    person_boxes.append((x1, y1, x2, y2))

        # Process each detected person
        for idx, (x1, y1, x2, y2) in enumerate(person_boxes):

            # Safety clamp
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)

            person_crop = frame[y1:y2, x1:x2]
            if person_crop.size == 0:
                continue

            rgb_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
            pose_result = pose.process(rgb_crop)

            if not pose_result.pose_landmarks:
                continue

            # Extract landmarks
            landmarks = []
            for lm in pose_result.pose_landmarks.landmark:
                px = int(lm.x * (x2 - x1)) + x1
                py = int(lm.y * (y2 - y1)) + y1
                landmarks.append([px, py])

            landmarks = np.array(landmarks)

            # Smooth landmarks
            smoothed = smooth_landmarks(prev_landmarks.get(idx), landmarks)
            prev_landmarks[idx] = smoothed

            # Draw skeleton
            for connection in mp_pose.POSE_CONNECTIONS:
                start = smoothed[connection[0]]
                end = smoothed[connection[1]]
                cv2.line(frame, tuple(start), tuple(end), (0, 255, 0), 2)

            for (x, y) in smoothed:
                cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)

        out.write(frame)

    cap.release()
    out.release()

    return output_path
