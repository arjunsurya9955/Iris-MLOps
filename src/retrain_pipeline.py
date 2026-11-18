
import mlflow
import mlflow.sklearn

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from src.monitor_drift import check_data_drift


# MLflow configuration
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "iris-continuous-retraining"

MODEL_NAME = "IrisRandomForestModel"

ACCURACY_THRESHOLD = 0.90


def train_and_register_model():

    print("Initiating retraining process...")

    # Step 1: Load Iris dataset
    iris = load_iris()

    X = iris.data
    y = iris.target

    # Step 2: Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Step 3: Configure Random Forest
    n_estimators = 100
    max_depth = 5

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )

    # Step 4: Train model
    model.fit(X_train, y_train)

    # Step 5: Evaluate model
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print(f"Candidate Model Accuracy: {accuracy:.4f}")

    # Step 6: Configure MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    mlflow.set_experiment(EXPERIMENT_NAME)

    # Step 7: Log and register model
    with mlflow.start_run(run_name="automated_retraining"):

        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("trigger", "data_drift")

        mlflow.log_metric("accuracy", accuracy)

        if accuracy >= ACCURACY_THRESHOLD:

            mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                registered_model_name=MODEL_NAME,
                skops_trusted_types=[
                    "sklearn.tree._tree.Tree"
                ]
            )

            print("Model successfully logged and registered.")
            print(f"Registered Model: {MODEL_NAME}")

        else:

            print("Accuracy threshold not met.")
            print("Model registration skipped.")

    return accuracy


def run_pipeline():

    print("Checking production data for drift...")

    drift_detected = check_data_drift()

    if drift_detected:

        print(
            "Data drift confirmed! "
            "Triggering retraining pipeline..."
        )

        train_and_register_model()

    else:

        print("No significant drift detected.")
        print("Retraining skipped.")


if __name__ == "__main__":

    run_pipeline()
