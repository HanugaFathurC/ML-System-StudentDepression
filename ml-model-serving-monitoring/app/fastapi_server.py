from fastapi.applications import FastAPI
from fastapi.responses import JSONResponse
from fastapi.requests import Request
import uuid
import json
import redis
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Redis connection
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

# Init FastAPI
app = FastAPI()
# Init Redis
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)

@app.post("/predict_queue")
async def predict_queue(request: Request):
    try:
        data = await request.json()
        request_id = str(uuid.uuid4())

        redis_client.lpush("inference_queue", json.dumps({
            "id": request_id,
            "input": data["input"]
        }))

        return JSONResponse(content={
            "status": "queued",
            "job_id": request_id
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/get_result/{request_id}")
async def get_result(request_id: str):
    try:
        key = f"result:{request_id}"
        # Fetch result from Redis
        print(f"Fetching result for job_id {request_id} from Redis with key: {key}")
        result = redis_client.get(key)
        print(f"Fetching result for job_id {request_id}: {result}")
        if result:
            return JSONResponse(content={
                "status": "completed",
                "job_id": request_id,
                "result": json.loads(result)}
            )
        else:
            return JSONResponse(content={"status": "pending", "job_id": request_id})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)