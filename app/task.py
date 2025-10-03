from typing import Dict
try:
    from app.main import analyze_with_yolo_and_tracking, analyze_video_naive, MODEL
except Exception:
    analyze_with_yolo_and_tracking = None
    analyze_video_naive = None
    MODEL = None

def analyze_file_job(video_path: str, use_yolo: bool = True) -> Dict:
    if use_yolo and analyze_with_yolo_and_tracking is not None and MODEL is not None:
        return analyze_with_yolo_and_tracking(video_path)
    else:
        return analyze_video_naive(video_path)