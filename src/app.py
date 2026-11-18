
import joblib
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "model" / "iris_model.joblib"

app = FastAPI(
    title="Iris MLOps API",
    description="Iris classification using a trained Random Forest model",
    version="1.0.0"
)

model = None

@app.on_event("startup")
def startup_event():
    global model

    try:
        model = joblib.load(MODEL_PATH)
        print("Iris model loaded successfully!")
    except Exception as error:
        print(f"Model loading failed: {error}")


class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.post("/predict")
def predict(data: IrisInput):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model artifact unavailable."
        )

    try:
        features = [[
            data.sepal_length,
            data.sepal_width,
            data.petal_length,
            data.petal_width
        ]]

        prediction = model.predict(features)

        return {
            "prediction": int(prediction[0])
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}"
        )
