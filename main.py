from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Request
from fastapi.responses import JSONResponse
import httpx
import tempfile
import os
import json

app = FastAPI()

@app.middleware("http")
async def cors_middleware(request: Request, call_next):
    response = await call_next(request)
    origin = request.headers.get("origin", "")
    response.headers["Access-Control-Allow-Origin"] = origin if origin else "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response

@app.options("/{rest_of_path:path}")
async def preflight(rest_of_path: str):
    return JSONResponse(content={}, headers={
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "*",
    })

@app.get("/")
def root():
    return {"status": "ok", "service": "Kaiwa Studio Backend"}

@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    openai_key: str = Form(...),
):
    if not openai_key:
        raise HTTPException(status_code=400, detail="openai_key is required")

    suffix = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            with open(tmp_path, "rb") as f:
                res = await client.post(
                    "https://api.openai.com/v1/audio/transcriptions",
                    headers={"Authorization": f"Bearer {openai_key}"},
                    files={"file": (file.filename or "audio.wav", f, "audio/wav")},
                    data={
                        "model": "whisper-1",
                        "language": "ja",
                        "response_format": "verbose_json",  # Get timestamps
                        "timestamp_granularities[]": "segment"
                    },
                )

        if res.status_code != 200:
            raise HTTPException(status_code=res.status_code, detail=res.text[:300])

        data = res.json()
        # Return both plain text and segments with timestamps
        segments = data.get("segments", [])
        transcript = data.get("text", "").strip()
        
        return {
            "transcript": transcript,
            "segments": [
                {
                    "text": s["text"].strip(),
                    "start": round(s["start"], 2),
                    "end": round(s["end"], 2)
                }
                for s in segments
            ]
        }

    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Whisper timeout")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try: os.unlink(tmp_path)
        except: pass

