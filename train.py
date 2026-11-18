
import mlflow
import mlflow.sklearn

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# MLflow configuration
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "iris-continuous-retraining"
MODEL_NAME = "IrisRandomForestModel"


def train_model():

    # Step 1: Load Iris dataset
    print("Loading Iris dataset...")

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

    # Step 3: Train Random Forest
    print("Training Random Forest model...")

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Step 4: Evaluate model
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print(f"Model Accuracy: {accuracy:.4f}")

    # Step 5: Configure MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    mlflow.set_experiment(EXPERIMENT_NAME)

    # Step 6: Log parameters, metrics and register model
    with mlflow.start_run(run_name="initial_training"):

        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 5)
        mlflow.log_param("random_state", 42)

        mlflow.log_metric("accuracy", accuracy)

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            registered_model_name=MODEL_NAME,
            skops_trusted_types=[
                "sklearn.tree._tree.Tree"
            ]
        )

    print("Model successfully logged to MLflow.")
    print(f"Registered Model: {MODEL_NAME}")
    print(f"Final Accuracy: {accuracy:.4f}")

    return model


if __name__ == "__main__":
    train_model()
