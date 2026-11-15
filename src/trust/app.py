import os
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from .scenes import build_all

app = FastAPI(title="Trust Layer demo (offline)")
_cache = {}


@app.get("/api/scenes")
def scenes():
    if "d" not in _cache:
        _cache["d"] = build_all()
    return JSONResponse(_cache["d"])


@app.get("/")
def index():
    return FileResponse(os.path.join(os.path.dirname(__file__), "static", "index.html"))
