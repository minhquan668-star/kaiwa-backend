from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import httpx
import tempfile
import os

app = FastAPI()

# Handle null origin from file:// protocol
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
                    data={"model": "whisper-1", "language": "ja", "response_format": "text"},
                )

        if res.status_code != 200:
            raise HTTPException(status_code=res.status_code, detail=res.text[:300])

        return {"transcript": res.text.strip()}

    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Whisper timeout — try a shorter clip")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try: os.unlink(tmp_path)
        except: pass
