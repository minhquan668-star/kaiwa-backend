from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import httpx
import tempfile
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_origin_regex=".*",
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)

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
