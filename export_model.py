
import mlflow
import mlflow.sklearn
import joblib

from pathlib import Path
from mlflow.tracking import MlflowClient

MODEL_NAME = "IrisRandomForestModel"

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()

versions = client.search_model_versions(
    f"name='{MODEL_NAME}'"
)

if not versions:
    raise RuntimeError("No registered model found.")

latest = max(versions, key=lambda v: int(v.version))

print(f"Exporting model version: {latest.version}")

model_uri = f"models:/{MODEL_NAME}/{latest.version}"
model = mlflow.sklearn.load_model(model_uri)

model_dir = Path("model")
model_dir.mkdir(exist_ok=True)

output_path = model_dir / "iris_model.joblib"
joblib.dump(model, output_path)

print("Model exported successfully!")
print(f"Saved at: {output_path.resolve()}")
