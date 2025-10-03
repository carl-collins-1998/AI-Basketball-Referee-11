"""Visualize YOLO detections or naive detections and the heuristic hoop line.
Saves an annotated output video so you can inspect the hoop_line and detections
to pick a better hoop_line_ratio.

Usage:
  python scripts/visualize_detections.py --video /path/to/input.mp4 --out /tmp/out.mp4 --model-path models/best.pt --sample-rate 3 --hoop-ratio 0.4

If ultralytics is not installed or model not provided, falls back to naive BG-subtraction.
"""
import argparse
import os
import cv2
import numpy as np

# Try to import ultralytics YOLO if available
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except Exception:
    ULTRALYTICS_AVAILABLE = False

def draw_hoop_line(frame, ratio, color=(0,255,0), thickness=2):
    h = frame.shape[0]
    y = int(h * ratio)
    cv2.line(frame, (0,y), (frame.shape[1], y), color, thickness)
    cv2.putText(frame, f'hoop_ratio={ratio:.2f}', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

def naive_detections(frame, backSub):
    fg = backSub.apply(frame)
    fg = cv2.medianBlur(fg, 5)
    _, thresh = cv2.threshold(fg, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dets = []
    if contours:
        for c in contours:
            area = cv2.contourArea(c)
            if area > 500:
                x,y,w,h = cv2.boundingRect(c)
                dets.append((x,y,x+w,y+h))
    return dets

def yolo_detections(frame, model, class_hint='ball'):
    results = model(frame)
    res0 = results[0]
    boxes = getattr(res0, 'boxes', None)
    dets = []
    if boxes is None:
        return dets
    # attempt to extract xyxy
    try:
        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, 'cpu') else np.array(boxes.xyxy)
        classes = boxes.cls.cpu().numpy() if hasattr(boxes.cls, 'cpu') else np.array(boxes.cls)
    except Exception:
        # fallback iteration
        for b in boxes:
            try:
                xy = b.xyxy[0]
                x1,y1,x2,y2 = int(xy[0]), int(xy[1]), int(xy[2]), int(xy[3])
                dets.append((x1,y1,x2,y2))
            except Exception:
                pass
        return dets
    # optionally filter class names
    model_names = None
    try:
        model_names = model.model.names
    except Exception:
        model_names = None
    for i, box in enumerate(xyxy):
        x1,y1,x2,y2 = [int(x) for x in box]
        if model_names is not None and class_hint is not None:
            cls = int(classes[i]) if i < len(classes) else None
            name = model_names.get(cls) if isinstance(model_names, dict) else (model_names[cls] if cls is not None and cls < len(model_names) else None)
            if name is not None and class_hint.lower() not in str(name).lower():
                continue
        dets.append((x1,y1,x2,y2))
    return dets

def annotate_frame(frame, dets, hoop_ratio):
    for (x1,y1,x2,y2) in dets:
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)
    draw_hoop_line(frame, hoop_ratio)
    return frame

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--video', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--model-path', default=None)
    p.add_argument('--sample-rate', type=int, default=3)
    p.add_argument('--hoop-ratio', type=float, default=0.4, help='fraction from top')
    p.add_argument('--class-hint', type=str, default='ball')
    args = p.parse_args()

    use_yolo = False
    model = None
    if args.model_path and ULTRALYTICS_AVAILABLE and os.path.exists(args.model_path):
        try:
            model = YOLO(args.model_path)
            use_yolo = True
            print('Using YOLO model for visualization.')
        except Exception as e:
            print('Failed to load YOLO model:', e)
            use_yolo = False

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit('Could not open input video')

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.out, fourcc, fps / max(1, args.sample_rate), (w,h))

    backSub = None
    if not use_yolo:
        backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % args.sample_rate == 0:
            if use_yolo:
                dets = yolo_detections(frame, model, args.class_hint)
            else:
                dets = naive_detections(frame, backSub)
            annotated = annotate_frame(frame.copy(), dets, args.hoop_ratio)
            out.write(annotated)
        idx += 1

    cap.release()
    out.release()
    print('Saved annotated output to', args.out)

if __name__ == '__main__':
    main()
