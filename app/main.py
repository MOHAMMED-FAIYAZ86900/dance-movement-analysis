from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import os
import uuid

from app.analyzer import analyze_video

app = FastAPI(title="Dance Movement Analysis API")

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


@app.post("/analyze")
async def analyze_dance_video(file: UploadFile = File(...)):
    """
    Upload a dance video and receive a skeleton-overlay video
    """

    # 1️⃣ Save uploaded video
    input_filename = f"{uuid.uuid4()}_{file.filename}"
    input_path = os.path.join(UPLOAD_DIR, input_filename)

    with open(input_path, "wb") as f:
        f.write(await file.read())

    # 2️⃣ Prepare output path
    output_filename = f"processed_{input_filename}"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    # 3️⃣ Run pose analysis
    analyze_video(input_path, output_path)

    # 4️⃣ Return processed video
    return FileResponse(
        path=output_path,
        media_type="video/mp4",
        filename=output_filename
    )
