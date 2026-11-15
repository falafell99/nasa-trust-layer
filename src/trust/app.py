import os
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from .scenes import build_all

app = FastAPI(title="Trust Layer demo (offline)")
_static = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=_static), name="static")
_cache = {}


@app.get("/api/scenes")
def scenes():
    if "d" not in _cache:
        _cache["d"] = build_all()
    return JSONResponse(_cache["d"])


@app.get("/")
def index():
    return FileResponse(os.path.join(os.path.dirname(__file__), "static", "index.html"))


@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)
