#!/bin/sh
# Одна команда: ./run_demo.sh  (из корня репозитория, без интернета) -> http://127.0.0.1:8000
exec .venv/bin/python -m uvicorn src.trust.app:app --host "${WATCH_DEMO_HOST:-127.0.0.1}" --port 8000
