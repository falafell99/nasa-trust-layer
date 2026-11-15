"""Local, short-lived Apple Health demo input. No raw Health export is retained."""
from __future__ import annotations

from datetime import datetime
from math import isfinite
from threading import Lock
from xml.etree.ElementTree import XMLPullParser, ParseError

from fastapi import HTTPException

TYPES = {
    "HKQuantityTypeIdentifierHeartRate": ("heart_rate", "bpm", 25, 240),
    "HKQuantityTypeIdentifierRespiratoryRate": ("respiratory_rate", "breaths/min", 4, 60),
    "HKQuantityTypeIdentifierOxygenSaturation": ("oxygen_saturation", "%", 0, 100),
    "HKQuantityTypeIdentifierAppleSleepingWristTemperature": ("wrist_temperature", "°C", 15, 45),
}
METRICS = {v[0]: v for v in TYPES.values()}
MAX_XML_BYTES = 200 * 1024 * 1024
_lock = Lock()
_samples: list[dict] = []


def _date(value: str) -> str:
    try:
        dt = datetime.strptime(value, "%Y-%m-%d %H:%M:%S %z")
    except ValueError:
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("Invalid sample timestamp") from exc
    if dt.tzinfo is None:
        raise ValueError("Timestamp needs a timezone")
    return dt.isoformat()


def _sample(metric: str, value: float, timestamp: str, source: str) -> dict:
    if metric not in METRICS or not isfinite(value):
        raise ValueError("Unsupported metric or non-finite value")
    _, unit, minimum, maximum = METRICS[metric]
    if not minimum <= value <= maximum:
        raise ValueError("Sample value outside accepted input range")
    return {"metric": metric, "value": round(value, 3), "unit": unit,
            "time": _date(timestamp), "source": source}


def latest() -> dict:
    with _lock:
        samples = list(_samples)
    return {"samples": samples, "source": "local Health import / iPhone Shortcut" if samples else None,
            "model_used": False}


def receive(items: list[dict]) -> dict:
    if not 1 <= len(items) <= 30:
        raise HTTPException(400, "Send 1–30 samples per request")
    try:
        clean = [_sample(str(x["metric"]), float(x["value"]), str(x["time"]),
                         "iPhone Shortcut (Health)") for x in items]
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc
    with _lock:
        _samples.extend(clean)
        _samples.sort(key=lambda x: x["time"])
        del _samples[:-240]
    return latest()


async def import_xml(chunks) -> dict:
    parser = XMLPullParser(events=("start", "end"))
    total = 0
    root = None
    found: list[dict] = []
    count = 0
    try:
        async for chunk in chunks:
            total += len(chunk)
            if total > MAX_XML_BYTES:
                raise HTTPException(413, "XML export exceeds 200 MB; use a smaller export or Shortcut")
            parser.feed(chunk)
            for event, elem in parser.read_events():
                if root is None and event == "start":
                    if elem.tag != "HealthData":
                        raise HTTPException(400, "Expected Apple Health export.xml")
                    root = elem
                if event != "end" or elem.tag != "Record":
                    continue
                kind = elem.attrib.get("type")
                watch = "apple watch" in (elem.attrib.get("sourceName", "") + " " + elem.attrib.get("device", "")).lower()
                if watch and kind in TYPES:
                    metric = TYPES[kind][0]
                    try:
                        value = float(elem.attrib["value"])
                        unit = elem.attrib.get("unit", "")
                        if metric == "oxygen_saturation" and value <= 1:
                            value *= 100
                        if metric == "wrist_temperature" and unit == "degF":
                            value = (value - 32) * 5 / 9
                        found.append(_sample(metric, value, elem.attrib["startDate"], "Apple Watch / Health export"))
                        count += 1
                        if len(found) > 500:
                            found.sort(key=lambda x: x["time"])
                            del found[:-240]
                    except (ValueError, KeyError):
                        pass
                elem.clear()
                if root is not None:
                    root.clear()
        parser.close()
    except ParseError as exc:
        raise HTTPException(400, "Invalid Apple Health XML") from exc
    if not found:
        raise HTTPException(422, "No supported Apple Watch samples found in this XML")
    found.sort(key=lambda x: x["time"])
    with _lock:
        _samples[:] = found[-240:]
    result = latest()
    result["imported_samples"] = count
    return result
