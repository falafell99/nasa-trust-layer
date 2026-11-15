"""V3: регуляризованная логистическая регрессия на признаках (features.py). Параметры хранятся в JSON (models/v3_logreg.json)."""
import json
import numpy as np
from .features import FEATURES
from .pipeline import Q_GOOD

T_RED = 0.5  # фиксирован заранее


def predict(params, f):
    x = np.array([f[k] for k in params["features"]], float)
    x = np.where(np.isnan(x), np.array(params["medians"]), x)
    z = (x - np.array(params["mean"])) / np.array(params["scale"])
    return float(1 / (1 + np.exp(-(z @ np.array(params["coef"]) + params["intercept"]))))


def decide_v3(q, p):
    """Два статуса: red только при q>=Q_GOOD и p>=T_RED; иначе yellow. Серого нет (отключён решением лида)."""
    return "red" if (q >= Q_GOOD and p >= T_RED) else "yellow"


def load(path="models/v3_logreg.json"):
    return json.load(open(path))
