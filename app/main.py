from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import uuid
import tempfile
import cv2
import numpy as np
from typing import Optional

app = FastAPI(title="AI Basketball Referee API")

# Allow CORS from anywhere for simplicity (adjust in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "/tmp/air_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/status")
async def status():
    return {"status": "ok"}

def analyze_video_naive(video_path: str, sample_rate: int = 3):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video file")

    centroid_ys = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    # Fallbacks
    if height == 0:
        height = 720
    if width == 0:
        width = 1280

    hoop_line_y = int(height * 0.4)

    backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_rate != 0:
            frame_idx += 1
            continue

        fg = backSub.apply(frame)
        fg = cv2.medianBlur(fg, 5)
        _, thresh = cv2.threshold(fg, 127, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(c)
            if area > 500:
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    centroid_ys.append((frame_idx, cY))
                else:
                    centroid_ys.append((frame_idx, None))
            else:
                centroid_ys.append((frame_idx, None))
        else:
            centroid_ys.append((frame_idx, None))

        frame_idx += 1

    cap.release()

    ys = [y for (_fi, y) in centroid_ys]
    made = 0
    missed = 0
    attempts = 0
    n = len(ys)
    look_ahead = max(5, int(15.0 / (sample_rate or 1)))

    for i in range(1, n):
        prev_y = ys[i - 1]
        cur_y = ys[i]
        if prev_y is not None and cur_y is not None:
            if prev_y < hoop_line_y and cur_y >= hoop_line_y:
                attempts += 1
                seen_after = False
                for j in range(i + 1, min(n, i + look_ahead + 1)):
                    if ys[j] is not None:
                        seen_after = True
                        break
                if not seen_after:
                    made += 1
                else:
                    missed += 1

    total = attempts
    accuracy = round(100.0 * made / total, 2) if total > 0 else 0.0

    return {
        "made_shots": int(made),
        "missed_shots": int(missed),
        "total_attempts": int(total),
        "accuracy": accuracy,
    }

@app.post("/analyze")
async def analyze(video: UploadFile = File(...)):
    suffix = os.path.splitext(video.filename)[1]
    tmp_name = f"{uuid.uuid4().hex}{suffix}"
    tmp_path = os.path.join(UPLOAD_DIR, tmp_name)

    try:
        with open(tmp_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {e}")
    finally:
        try:
            video.file.close()
        except:
            pass

    try:
        result = analyze_video_naive(tmp_path)
    except Exception as e:
        try:
            os.remove(tmp_path)
        except:
            pass
        raise HTTPException(status_code=500, detail=str(e))

    try:
        os.remove(tmp_path)
    except:
        pass

    return JSONResponse(content=result)