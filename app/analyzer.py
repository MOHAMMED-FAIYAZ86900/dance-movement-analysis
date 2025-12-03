import cv2
import mediapipe as mp
import os

# Initialize MediaPipe pose utilities
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def analyze_video(input_video_path: str, output_video_path: str) -> str:
    """
    Reads a video, detects human pose, draws skeleton, and saves output video.
    """

    # 1) Check if input video exists
    if not os.path.exists(input_video_path):
        raise FileNotFoundError(f"Input video not found: {input_video_path}")

    # 2) Open input video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError("Failed to open input video")

    # 3) Get video properties (fps, width, height)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 4) Prepare output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(
        output_video_path,
        fourcc,
        fps,
        (width, height)
    )

    # 5) Initialize Pose model
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        # 6) Read video frame by frame
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # No more frames

            # OpenCV uses BGR, MediaPipe expects RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 7) Detect pose
            results = pose.process(rgb_frame)

            # 8) If any person is detected, draw skeleton on the frame
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,                       # draw ON this frame
                    results.pose_landmarks,      # pose keypoints
                    mp_pose.POSE_CONNECTIONS     # how keypoints connect (skeleton)
                )

            # 9) Save the processed frame to output video
            out.write(frame)

    # 10) Release video resources
    cap.release()
    out.release()

    return output_video_path

# if __name__ == "__main__":
#     input_video = "sample_dance.mp4"         # must exist in project root
#     output_video = "output_skeleton.mp4"     # will be created

#     analyze_video(input_video, output_video)
#     print("✅ Skeleton video generated:", output_video)
