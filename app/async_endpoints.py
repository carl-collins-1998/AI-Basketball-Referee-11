from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import os
import uuid
import shutil
from redis import Redis
from rq import Queue
from app.tasks import analyze_file_job

router = APIRouter()
redis_url = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
redis_conn = Redis.from_url(redis_url)
q = Queue('default', connection=redis_conn)

UPLOAD_DIR = os.environ.get('UPLOAD_DIR', '/tmp/air_uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post('/analyze_async')
async def analyze_async(video: UploadFile = File(...), use_yolo: bool = True):
    suffix = os.path.splitext(video.filename)[1]
    tmp_name = f"{uuid.uuid4().hex}{suffix}"
    tmp_path = os.path.join(UPLOAD_DIR, tmp_name)
    try:
        with open(tmp_path, 'wb') as f:
            shutil.copyfileobj(video.file, f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            video.file.close()
        except:
            pass

    job = q.enqueue(analyze_file_job, tmp_path, use_yolo)
    return JSONResponse({'job_id': job.get_id(), 'status': 'queued'})

@router.get('/jobs/{job_id}')
def job_status(job_id: str):
    job = q.fetch_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail='Job not found')
    return JSONResponse({
        'id': job.get_id(),
        'status': job.get_status(),
        'result': job.result if job.is_finished else None,
        'exc_info': job.exc_info
    })
