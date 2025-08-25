import os
import sys
import uuid
import asyncio
import signal
import time
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional
import tempfile
import shutil
from contextlib import asynccontextmanager
import zipfile

from fastapi import FastAPI, UploadFile, File, HTTPException, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.openapi.docs import (
    get_redoc_html,
    get_swagger_ui_html,
    get_swagger_ui_oauth2_redirect_html,
)

import cv2
from basketball_referee import ImprovedFreeThrowScorer, CVATDatasetConverter, FreeThrowModelTrainer

# Configuration
MODEL_PATH = str(Path(__file__).parent / "models" / "best.pt")
PORT = int(os.getenv('PORT', 8000))
ENVIRONMENT = os.getenv('ENVIRONMENT', 'production')
MAX_UPLOAD_SIZE = int(os.getenv('MAX_UPLOAD_SIZE', 500))  # MB

# Keep-alive mechanism
keep_running = True


def handle_exit(signum, frame):
    global keep_running
    print(f"\nReceived signal {signum}, initiating graceful shutdown...")
    keep_running = False


def keep_alive():
    """Background thread to keep the application alive"""
    while keep_running:
        time.sleep(5)
        print("Keep-alive heartbeat", flush=True)


scorer_instance = None
training_jobs = {}


def is_valid_cvat_dataset(directory_path):
    """Check if the directory contains a valid CVAT dataset"""
    directory = Path(directory_path)

    # Check for required files/directories in a CVAT dataset
    required_items = [
        directory / 'obj_train_data',  # Contains images and labels
    ]

    # Check if at least one image file exists
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    has_images = any(
        file.suffix.lower() in image_extensions
        for file in (directory / 'obj_train_data').rglob('*')
        if file.is_file()
    )

    return all(item.exists() for item in required_items) and has_images


async def _train_model_background(job_id, dataset_files, epochs, batch_size, model_size, device):
    """Background task for model training"""
    global training_jobs, scorer_instance, MODEL_PATH

    training_jobs[job_id] = {
        "status": "running",
        "progress": 0,
        "message": "Starting training process..."
    }

    try:
        with tempfile.TemporaryDirectory() as temp_dataset_dir:
            dataset_paths = []

            # Process uploaded dataset files
            for i, dataset_file in enumerate(dataset_files):
                # Create a directory for this dataset
                dataset_dir = Path(temp_dataset_dir) / f"dataset_{i}"
                dataset_dir.mkdir(parents=True, exist_ok=True)

                # Save the uploaded file
                file_path = dataset_dir / dataset_file.filename
                content = await dataset_file.read()
                with open(file_path, "wb") as f:
                    f.write(content)

                # If it's a zip file, extract it
                if dataset_file.filename.lower().endswith('.zip'):
                    with zipfile.ZipFile(file_path, 'r') as zip_ref:
                        zip_ref.extractall(dataset_dir)
                    os.remove(file_path)  # Remove the zip file after extraction

                dataset_paths.append(str(dataset_dir))

                training_jobs[job_id]["progress"] = (i + 1) / len(dataset_files) * 10
                training_jobs[job_id]["message"] = f"Processed dataset {i + 1}/{len(dataset_files)}"

            # Check if we have valid CVAT datasets
            valid_datasets = []
            for dataset_path in dataset_paths:
                if is_valid_cvat_dataset(dataset_path):
                    valid_datasets.append(dataset_path)
                else:
                    print(f"Warning: {dataset_path} is not a valid CVAT dataset")

            if not valid_datasets:
                raise ValueError("No valid CVAT datasets found in uploaded files")

            # Convert CVAT to YOLO format
            training_jobs[job_id]["progress"] = 15
            training_jobs[job_id]["message"] = "Converting datasets to YOLO format..."

            converter = CVATDatasetConverter(valid_datasets, str(temp_dataset_dir))
            converter.convert_multiple_cvat_to_yolo()

            # Train the model
            training_jobs[job_id]["progress"] = 30
            training_jobs[job_id]["message"] = "Starting model training..."

            trainer = FreeThrowModelTrainer(str(temp_dataset_dir), model_size=model_size)

            # Train with progress updates (simplified)
            training_results = trainer.train_model(
                epochs=epochs,
                batch_size=batch_size,
                device=device
            )

            training_jobs[job_id]["progress"] = 80
            training_jobs[job_id]["message"] = "Validating model..."

            # Validate
            validation_metrics = trainer.validate_model()

            # Get the best model path
            trained_model_dir = Path("freethrow_training") / f"freethrow_yolov8{model_size}"
            best_model_paths = list(trained_model_dir.rglob("best.pt"))

            if not best_model_paths:
                raise ValueError("No best.pt file found after training")

            best_model_path = best_model_paths[0]

            # Copy the new model
            training_jobs[job_id]["progress"] = 90
            training_jobs[job_id]["message"] = "Saving trained model..."

            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

            if os.path.exists(MODEL_PATH):
                backup_path = MODEL_PATH + ".backup"
                shutil.copy2(MODEL_PATH, backup_path)

            shutil.copy2(best_model_path, MODEL_PATH)

            # Reload the scorer
            training_jobs[job_id]["progress"] = 95
            training_jobs[job_id]["message"] = "Loading new model..."

            try:
                scorer_instance = ImprovedFreeThrowScorer(MODEL_PATH)
            except Exception as e:
                print(f"Warning: Could not reload scorer with new model: {e}")

            # Prepare results
            training_jobs[job_id]["status"] = "completed"
            training_jobs[job_id]["progress"] = 100
            training_jobs[job_id]["message"] = "Training completed successfully!"
            training_jobs[job_id]["results"] = {
                "model_path": str(MODEL_PATH),
                "model_size": model_size,
                "epochs_trained": epochs,
                "batch_size": batch_size,
                "device_used": device,
                "datasets_used": len(valid_datasets),
                "validation_metrics": {
                    "mAP50": float(validation_metrics.box.map50),
                    "mAP50-95": float(validation_metrics.box.map),
                }
            }

    except Exception as e:
        training_jobs[job_id]["status"] = "failed"
        training_jobs[job_id]["message"] = f"Training failed: {str(e)}"
        print(f"Training job {job_id} failed: {e}")
        import traceback
        traceback.print_exc()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Enhanced lifespan with proper Railway handling."""
    global scorer_instance, keep_running

    # Setup signal handlers
    signal.signal(signal.SIGTERM, handle_exit)
    signal.signal(signal.SIGINT, handle_exit)

    # Start keep-alive thread
    threading.Thread(target=keep_alive, daemon=True).start()

    print("\n" + "=" * 60)
    print("AI BASKETBALL REFEREE API STARTING")
    print("=" * 60)
    print(f"Environment: {ENVIRONMENT}")
    print(f"Python version: {sys.version}")
    print(f"Model path: {MODEL_PATH}")
    print(f"Server port: {PORT}")

    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    if os.path.exists(MODEL_PATH):
        model_size = os.path.getsize(MODEL_PATH) / 1024 / 1024
        print(f"Model found! Size: {model_size:.2f} MB")
        try:
            print("Loading model...")
            scorer_instance = ImprovedFreeThrowScorer(MODEL_PATH)
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
    else:
        print("⚠️ Model file not found! Training required.")

    print("=" * 60 + "\n")

    yield

    print("\nInitiating graceful shutdown...")
    keep_running = False


app = FastAPI(
    title="AI Basketball Referee API",
    version="1.0.0",
    description="Automated basketball free throw detection and scoring",
    docs_url=None,  # Disable default docs
    redoc_url=None,  # Disable default redoc
    lifespan=lifespan
)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Custom docs endpoints
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - Swagger UI",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_js_url="/static/js/swagger-ui-bundle.js",
        swagger_css_url="/static/css/swagger-ui.css",
    )


@app.get(app.swagger_ui_oauth2_redirect_url, include_in_schema=False)
async def swagger_ui_redirect():
    return get_swagger_ui_oauth2_redirect_html()


@app.get("/redoc", include_in_schema=False)
async def redoc_html():
    return get_redoc_html(
        openapi_url=app.openapi_url,
        title=app.title + " - ReDoc",
        redoc_js_url="/static/js/redoc.standalone.js",
    )


@app.get("/", response_class=HTMLResponse)
async def serve_web_interface():
    return FileResponse('static/index.html')


@app.get("/health")
async def health():
    """Health check endpoint for monitoring."""
    return {"status": "healthy", "model_loaded": scorer_instance is not None}


@app.get("/model_status")
async def model_status():
    """Detailed model status."""
    return {
        "loaded": scorer_instance is not None,
        "path": MODEL_PATH,
        "exists": os.path.exists(MODEL_PATH),
        "size_mb": os.path.getsize(MODEL_PATH) / 1024 / 1024 if os.path.exists(MODEL_PATH) else 0,
        "environment": ENVIRONMENT
    }


@app.post("/upload_model/")
async def upload_model(model_file: UploadFile = File(...)):
    """Upload a model file to use for inference"""
    global scorer_instance

    if not model_file.filename.endswith('.pt'):
        raise HTTPException(status_code=400, detail="Model file must be a .pt file")

    # Check file size
    content = await model_file.read()
    file_size_mb = len(content) / 1024 / 1024

    if file_size_mb > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_UPLOAD_SIZE}MB"
        )

    print(f"Uploading model: {model_file.filename} ({file_size_mb:.2f} MB)")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    with open(MODEL_PATH, "wb") as f:
        f.write(content)

    print(f"Model saved to {MODEL_PATH}")

    try:
        scorer_instance = ImprovedFreeThrowScorer(MODEL_PATH)
        print("✅ New model loaded successfully!")
        return {
            "status": "success",
            "message": "Model uploaded and loaded successfully",
            "model_path": MODEL_PATH,
            "size_mb": file_size_mb
        }
    except Exception as e:
        os.remove(MODEL_PATH)
        print(f"❌ Failed to load uploaded model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


@app.post("/score_video/")
async def score_video(video_file: UploadFile = File(...)) -> Dict[str, Any]:
    """Analyzes an uploaded video to detect and score free throws."""
    global scorer_instance

    print(f"\n=== Processing video: {video_file.filename} ===")

    if scorer_instance is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please upload a model using /upload_model/"
        )

    # Check file size
    content = await video_file.read()
    file_size_mb = len(content) / 1024 / 1024

    if file_size_mb > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_UPLOAD_SIZE}MB"
        )

    with tempfile.TemporaryDirectory() as temp_dir:
        video_path = Path(temp_dir) / video_file.filename

        with open(video_path, "wb") as f:
            f.write(content)
        print(f"Video saved: {file_size_mb:.2f} MB)")

        # Reset scorer
        scorer_instance.made_shots = 0
        scorer_instance.missed_shots = 0
        scorer_instance.shot_attempts = 0
        scorer_instance.shot_tracker.reset()

        # Process video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Could not open video file")

        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Total frames: {total_frames}, FPS: {fps:.2f}")

        start_time = time.time()

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                # Run detection
                detections = scorer_instance.detect_objects(frame)
                hoop_info = scorer_instance.update_hoop_position(detections)
                ball_info = scorer_instance.find_ball(detections)
                player_bboxes = scorer_instance.find_players(detections)

                # Update shot tracking
                old_phase = scorer_instance.shot_tracker.shot_phase
                result = scorer_instance.shot_tracker.update(ball_info, hoop_info, player_bboxes, False)

                # Count attempts
                if old_phase == 'idle' and scorer_instance.shot_tracker.shot_phase == 'rising':
                    scorer_instance.shot_attempts += 1
                    print(f"Shot attempt #{scorer_instance.shot_attempts} at frame {frame_count}")

                # Count results
                if result == 'score':
                    scorer_instance.made_shots += 1
                    print(f"SCORE! Total: {scorer_instance.made_shots}")
                    scorer_instance.shot_tracker.reset()
                elif result == 'miss':
                    scorer_instance.missed_shots += 1
                    print(f"MISS! Total: {scorer_instance.missed_shots}")
                    scorer_instance.shot_tracker.reset()

                if frame_count % 100 == 0:
                    print(f"Progress: {frame_count}/{total_frames} frames ({frame_count / total_frames * 100:.1f}%)")

        except Exception as e:
            print(f"Error during video processing: {e}")
            cap.release()
            raise HTTPException(status_code=500, detail=f"Video processing error: {str(e)}")

        cap.release()

        print(f"Processing complete. Frames: {frame_count}")

        accuracy = (
                scorer_instance.made_shots / scorer_instance.shot_attempts * 100
        ) if scorer_instance.shot_attempts > 0 else 0

        return {
            "made_shots": scorer_instance.made_shots,
            "missed_shots": scorer_instance.missed_shots,
            "total_attempts": scorer_instance.shot_attempts,
            "accuracy_percentage": round(accuracy, 1),
            "frames_processed": frame_count,
            "video_duration_seconds": round(total_frames / fps, 2) if fps > 0 else 0,
            "processing_time_per_frame": round(time.time() - start_time, 4) / frame_count if frame_count > 0 else 0
        }


@app.post("/train_model/")
async def train_model(
        background_tasks: BackgroundTasks,
        dataset_files: List[UploadFile] = File(..., description="CVAT dataset files (zip files or unzipped folders)"),
        epochs: int = Form(150),
        batch_size: int = Form(16),
        model_size: str = Form("s"),
        device: str = Form("auto")
) -> Dict[str, Any]:
    """Trains a new basketball referee model using uploaded CVAT annotated datasets."""
    # Validate inputs
    if model_size not in ["n", "s", "m", "l"]:
        raise HTTPException(status_code=400, detail="Model size must be n, s, m, or l")

    if device not in ["cpu", "cuda", "auto"]:
        raise HTTPException(status_code=400, detail="Device must be cpu, cuda, or auto")

    if not dataset_files:
        raise HTTPException(status_code=400, detail="At least one dataset file is required")

    print(f"\n=== Training New Model ===")
    print(f"Datasets: {len(dataset_files)} files")
    print(f"Epochs: {epochs}, Batch size: {batch_size}, Model size: {model_size}, Device: {device}")

    # Generate job ID
    job_id = str(uuid.uuid4())

    # Start training in background
    background_tasks.add_task(
        _train_model_background,
        job_id,
        dataset_files,
        epochs,
        batch_size,
        model_size,
        device
    )

    return {
        "status": "started",
        "job_id": job_id,
        "message": "Training started in background",
        "check_status_url": f"/training_status/{job_id}"
    }


@app.get("/training_status/{job_id}")
async def training_status(job_id: str):
    """Check the status of a training job."""
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    return training_jobs[job_id]


@app.get("/training_jobs")
async def list_training_jobs():
    """List all training jobs."""
    return {
        "jobs": list(training_jobs.keys()),
        "active_jobs": [job_id for job_id, job in training_jobs.items() if job["status"] == "running"],
        "completed_jobs": [job_id for job_id, job in training_jobs.items() if job["status"] == "completed"],
        "failed_jobs": [job_id for job_id, job in training_jobs.items() if job["status"] == "failed"]
    }


@app.delete("/training_jobs/{job_id}")
async def delete_training_job(job_id: str):
    """Delete a training job record."""
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    del training_jobs[job_id]
    return {"status": "deleted", "job_id": job_id}


if __name__ == "__main__":
    import uvicorn

    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=PORT,
        log_level="info",
        timeout_keep_alive=300,
        limit_max_requests=10000,
        workers=1
    )

    server = uvicorn.Server(config)

    try:
        print(f"\n🚀 Starting server on port {PORT}")
        server.run()
    except Exception as e:
        print(f"Server error: {e}")
    finally:
        print("Server shutdown complete")