🩰 Dance Movement Analysis API (Production-Grade)

A production-ready multi-person dance movement analysis system that detects dancers in a video and generates stable skeleton overlay videos using a detection-first pipeline.
The system is optimized for CPU-only cloud environments and deployed using Docker on Google Cloud Platform (GCP).

📌 Project Overview

This project analyzes dance videos and extracts human skeletal movements for single or multiple dancers, even in challenging scenarios such as:

Multiple people in the frame

Occlusions and overlaps

Fast dance movements

The system is designed following industry-grade ML system architecture, ensuring accuracy, stability, and scalability.

🧠 System Architecture
Input Video
   ↓
YOLOv8 (Person Detection – every N frames)
   ↓
OpenCV Tracker (Person Identity Tracking)
   ↓
MediaPipe Pose (Skeleton Estimation per Person)
   ↓
Temporal Smoothing (Stable Joints)
   ↓
Skeleton Overlay Video (Output)

🚀 Key Features

✅ Multi-person pose estimation

✅ Stable skeleton tracking (no jitter)

✅ Identity preservation using trackers

✅ CPU-optimized (no GPU required)

✅ FastAPI REST API

✅ Dockerized deployment

✅ Cloud-ready (GCP Compute Engine)

✅ Swagger UI for easy testing

🔍 Why Detection-First Pipeline?

MediaPipe Pose works best for single-person scenarios.
In real-world dance videos with multiple dancers, it can produce incorrect or unstable skeletons.

To solve this, the system uses:

YOLOv8 → Detect all people in the frame

OpenCV Trackers → Maintain consistent person IDs

MediaPipe Pose → Estimate skeleton per individual

This approach ensures correct, stable, and scalable pose estimation.

🛠️ Tech Stack

Programming Language: Python

Person Detection: YOLOv8 (Ultralytics)

Pose Estimation: MediaPipe Pose

Tracking: OpenCV KCF Tracker

Backend: FastAPI

Containerization: Docker

Cloud Platform: Google Cloud Platform (GCP)

Inference: CPU-only (cost-optimized)

📦 API Details
Endpoint
POST /analyze

Request

multipart/form-data

Upload a dance video file

Response

Processed video with skeleton overlays

Swagger UI
http://<VM-IP>:8000/docs

☁️ Deployment Details

Deployed on GCP Compute Engine

Uses persistent disk storage for ML dependencies

Docker storage migrated to external disk for reliability

CPU-optimized inference to reduce cost

🧪 How to Run Locally (Optional)
git clone https://github.com/MOHAMMED-FAIYAZ86900/dance-movement-analysis.git
cd dance-movement-analysis

docker build -t dance-analysis .
docker run -p 8000:8000 dance-analysis


Open:

http://localhost:8000/docs

🎯 Challenges Solved

✔ Dependency conflicts (MediaPipe, Torch, OpenCV)

✔ Disk limitations on cloud VMs

✔ Multi-person pose instability

✔ CPU inference optimization

✔ Production-grade Docker deployment

🎤 Interview-Ready Summary

“I designed a production-grade multi-person dance analysis system using a detection-first architecture. YOLOv8 detects dancers, OpenCV trackers preserve identity, and MediaPipe Pose estimates skeletons per person. The system is optimized for CPU inference, containerized with Docker, and deployed on Google Cloud Platform.”

👤 Author

Mohammed Faiyaz
Artificial Intelligence & Machine Learning Engineer
GitHub: https://github.com/MOHAMMED-FAIYAZ86900

⭐ Final Note

This project demonstrates real-world ML system design, cloud deployment, and performance optimization, making it suitable for:

ML / AI Internships

Backend / ML Engineer roles

Academic projects & demos

Portfolio showcases

If you want, next I can help you with:

✅ Resume bullet points

✅ LinkedIn project post

✅ PPT / project report

✅ Mock interview Q&A

Just tell me 👍
