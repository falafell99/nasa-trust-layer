import os
import secrets
import json
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from .scenes import build_all
from . import wearable

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
def index(request: Request):
    name = "index.html" if "scene" in request.query_params else "wearable.html"
    return FileResponse(os.path.join(_static, name))


@app.get("/cases")
def cases():
    return FileResponse(os.path.join(_static, "index.html"))


@app.get("/api/watch/latest")
def watch_latest(request: Request):
    local = request.client and request.client.host in ("127.0.0.1", "::1")
    token = os.environ.get("WATCH_DEMO_TOKEN")
    authorized = token and secrets.compare_digest(request.headers.get("x-watch-demo-token", ""), token)
    if not local and not authorized:
        raise HTTPException(401, "Local access or demo token required")
    return wearable.latest()


@app.post("/api/watch/import")
async def watch_import(request: Request):
    if not request.client or request.client.host not in ("127.0.0.1", "::1"):
        raise HTTPException(403, "Health export can only be imported on this computer")
    if request.headers.get("content-type", "").split(";")[0] not in ("application/xml", "text/xml"):
        raise HTTPException(415, "Send the extracted export.xml file")
    return await wearable.import_xml(request.stream())


@app.post("/api/watch/samples")
async def watch_samples(request: Request):
    token = os.environ.get("WATCH_DEMO_TOKEN")
    if not token:
        raise HTTPException(503, "Shortcut receiver is disabled; set WATCH_DEMO_TOKEN")
    if not secrets.compare_digest(request.headers.get("x-watch-demo-token", ""), token):
        raise HTTPException(401, "Invalid demo token")
    chunks = bytearray()
    async for chunk in request.stream():
        chunks.extend(chunk)
        if len(chunks) > 16_384:
            raise HTTPException(413, "Shortcut request is too large")
    try:
        body = json.loads(chunks)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise HTTPException(400, "Invalid JSON") from exc
    if not isinstance(body, dict):
        raise HTTPException(400, "Expected JSON object")
    return wearable.receive(body.get("samples", []))


@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)
