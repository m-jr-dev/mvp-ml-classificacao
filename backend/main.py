
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ConfigDict
from starlette.templating import Jinja2Templates

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "wine_classifier.joblib"
METADATA_PATH = ARTIFACTS_DIR / "model_metadata.json"

FEATURE_COLUMNS = [
    "alcohol",
    "malic_acid",
    "ash",
    "alcalinity_of_ash",
    "magnesium",
    "total_phenols",
    "flavanoids",
    "nonflavanoid_phenols",
    "proanthocyanins",
    "color_intensity",
    "hue",
    "od280_od315_of_diluted_wines",
    "proline",
]

app = FastAPI(title="Wine Classifier MVP", version="1.0.0")

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

model = joblib.load(MODEL_PATH)
metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))


class PredictionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    alcohol: float = Field(..., ge=0)
    malic_acid: float = Field(..., ge=0)
    ash: float = Field(..., ge=0)
    alcalinity_of_ash: float = Field(..., ge=0)
    magnesium: float = Field(..., ge=0)
    total_phenols: float = Field(..., ge=0)
    flavanoids: float = Field(..., ge=0)
    nonflavanoid_phenols: float = Field(..., ge=0)
    proanthocyanins: float = Field(..., ge=0)
    color_intensity: float = Field(..., ge=0)
    hue: float = Field(..., ge=0)
    od280_od315_of_diluted_wines: float = Field(..., ge=0)
    proline: float = Field(..., ge=0)


TARGET_LABELS = {
    1: "Classe 1",
    2: "Classe 2",
    3: "Classe 3",
}


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "request": request,
            "selected_model": metadata["selected_model"],
            "metrics": metadata["selected_model_metrics"],
        },
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "selected_model": metadata["selected_model"],
    }


@app.get("/model-info")
def model_info():
    return metadata


@app.post("/predict")
def predict(payload: PredictionInput):
    row = pd.DataFrame([{column: getattr(payload, column) for column in FEATURE_COLUMNS}])

    try:
        prediction = int(model.predict(row)[0])
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Falha ao processar a predição: {exc}") from exc

    return {
        "prediction": prediction,
        "label": TARGET_LABELS.get(prediction, f"Classe {prediction}"),
        "selected_model": metadata["selected_model"],
    }
