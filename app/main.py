from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import shutil
import os
import uuid
import math
import cv2
import numpy as np

# Try to import YOLO from ultralytics, set flag if available
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except Exception:
    ULTRALYTICS_AVAILABLE = False

app = FastAPI(title="AI Basketball Referee API")

# Allow CORS from anywhere for simplicity (adjust in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (UI) from /static and index at /
app.mount("/static", StaticFiles(directory="static"), name="static")

# Optional configuration via env vars
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "/tmp/air_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

MODEL_PATH = os.environ.get("MODEL_PATH", "models/best.pt")
MODEL_USE = os.environ.get("MODEL_USE", "yolo")  # 'yolo' or 'naive'
HOOP_LINE_RATIO = float(os.environ.get("HOOP_LINE_RATIO", "0.4"))

# If ultralytics is available and model file exists, load it at startup
MODEL = None
if ULTRALYTICS_AVAILABLE and os.path.exists(MODEL_PATH) and MODEL_USE == 'yolo':
    try:
        MODEL = YOLO(MODEL_PATH)
    except Exception:
        MODEL = None

# Include async router if available
try:
    from app.async_endpoints import router as async_router
    app.include_router(async_router)
except Exception:
    pass

@app.get("/")
async def root():
    return FileResponse("static/index.html")

@app.get("/status")
async def status():
    return {"status": "ok", "yolo_available": ULTRALYTICS_AVAILABLE, "model_loaded": MODEL is not None}

# ----------------------- Helper: YOLO + simple tracker based analyzer -----------------------

def analyze_with_yolo_and_tracking(video_path: str, sample_rate: int = 3, class_name_hint: str = 'ball'):
    if MODEL is None:
        raise RuntimeError('YOLO model is not loaded')

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video file")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_FRAME_HEIGHT) or 0) if False else None
    # Use correct property for height
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if height == 0:
        height = 720
    if width == 0:
        width = 1280

    hoop_line_y = int(height * HOOP_LINE_RATIO)

    tracks = {}
    next_track_id = 1
    max_distance = max(50, int(min(width, height) * 0.08))
    disappear_tolerance = 8

    frame_idx = 0
    attempts = 0
    made = 0
    missed = 0

    # Map model class ids to names if available
    model_names = None
    try:
        model_names = MODEL.model.names if hasattr(MODEL, 'model') and hasattr(MODEL.model, 'names') else None
    except Exception:
        model_names = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_rate != 0:
            frame_idx += 1
            continue

        try:
            results = MODEL(frame)
        except Exception:
            frame_idx += 1
            continue

        detections = []
        res0 = results[0]
        boxes = getattr(res0, 'boxes', None)
        if boxes is not None:
            try:
                xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, 'cpu') else np.array(boxes.xyxy)
                confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, 'cpu') else np.array(boxes.conf)
                classes = boxes.cls.cpu().numpy() if hasattr(boxes.cls, 'cpu') else np.array(boxes.cls)
            except Exception:
                xyxy = []
                confs = []
                classes = []
                for b in boxes:
                    try:
                        xy = b.xyxy[0]
                        xyxy.append([xy[0], xy[1], xy[2], xy[3]])
                        confs.append(float(b.conf[0]))
                        classes.append(int(b.cls[0]))
                    except Exception:
                        pass
                xyxy = np.array(xyxy)
                confs = np.array(confs)
                classes = np.array(classes)

            for i, box in enumerate(xyxy):
                conf = float(confs[i]) if i < len(confs) else 0.0
                cls = int(classes[i]) if i < len(classes) else 0
                if model_names is not None and class_name_hint is not None:
                    name = model_names.get(cls) if isinstance(model_names, dict) else (model_names[cls] if cls < len(model_names) else None)
                    if name is not None and class_name_hint.lower() not in str(name).lower():
                        continue
                x1, y1, x2, y2 = box
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                detections.append((cx, cy, conf))

        for tid, tr in list(tracks.items()):
            tr['assigned'] = False

        for det_idx, (cx, cy, _) in enumerate(detections):
            best_id = None
            best_dist = None
            for tid, tr in tracks.items():
                lx, ly = tr['centroid']
                dist = math.hypot(cx - lx, cy - ly)
                if dist <= max_distance and (best_dist is None or dist < best_dist):
                    best_dist = dist
                    best_id = tid
            if best_id is not None:
                tracks[best_id]['centroid'] = (cx, cy)
                tracks[best_id]['ys'].append((frame_idx, cy))
                tracks[best_id]['last_seen_frame'] = frame_idx
                tracks[best_id]['disappeared_count'] = 0
                tracks[best_id]['assigned'] = True
            else:
                tracks[next_track_id] = {
                    'centroid': (cx, cy),
                    'ys': [(frame_idx, cy)],
                    'last_seen_frame': frame_idx,
                    'disappeared_count': 0,
                    'scored': False,
                }
                next_track_id += 1

        to_delete = []
        for tid, tr in list(tracks.items()):
            if not tr.get('assigned', False):
                tr['disappeared_count'] = tr.get('disappeared_count', 0) + 1
            ys = [y for (_f, y) in tr['ys']]
            if not tr.get('scored', False) and len(ys) >= 2:
                for i in range(1, len(ys)):
                    prev_y = ys[i-1]
                    cur_y = ys[i]
                    if prev_y is not None and cur_y is not None and prev_y < hoop_line_y and cur_y >= hoop_line_y:
                        attempts += 1
                        tr['crossed_at_frame'] = tr['ys'][i][0]
                        tr['scored'] = 'pending'
                        break

            if tr.get('scored') == 'pending' and tr.get('disappeared_count', 0) >= disappear_tolerance:
                made += 1
                tr['scored'] = True

            if tr.get('disappeared_count', 0) > (disappear_tolerance * 10):
                to_delete.append(tid)

        for tid in to_delete:
            del tracks[tid]

        frame_idx += 1

    for tid, tr in tracks.items():
        if tr.get('scored') == 'pending':
            missed += 1

    cap.release()

    total = attempts
    accuracy = round(100.0 * made / total, 2) if total > 0 else 0.0
    return {"made_shots": int(made), "missed_shots": int(missed), "total_attempts": int(total), "accuracy": accuracy}

# ----------------------- Fallback naive analyzer (existing) -----------------------

def analyze_video_naive(video_path: str, sample_rate: int = 3):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video file")

    centroid_ys = []
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    if height == 0:
        height = 720
    if width == 0:
        width = 1280

    hoop_line_y = int(height * HOOP_LINE_RATIO)

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

    return {"made_shots": int(made), "missed_shots": int(missed), "total_attempts": int(total), "accuracy": accuracy}

# ----------------------- API endpoint -----------------------
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
        if MODEL is not None and MODEL_USE == 'yolo':
            result = analyze_with_yolo_and_tracking(tmp_path)
        else:
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
