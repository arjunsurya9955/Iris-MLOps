
import mlflow
from mlflow.tracking import MlflowClient
from pathlib import Path

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()

versions = client.search_model_versions(
    "name='IrisRandomForestModel'"
)

latest = max(versions, key=lambda v: int(v.version))

print("Latest model version:", latest.version)
print("Model ID:", latest.source)

logged_model = client.get_logged_model(
    latest.source.split("/")[-1]
)

print("Artifact location:", logged_model.artifact_location)

print("\nProject folders:")
for path in Path(".").iterdir():
    print(path.name, "(folder)" if path.is_dir() else "(file)")
