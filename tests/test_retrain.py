
import mlflow

from src.monitor_drift import check_data_drift

from src.retrain_pipeline import (
    train_and_register_model,
    run_pipeline,
    MODEL_NAME
)


def test_data_drift_detection():
    """Verify that drift detection returns a Boolean."""

    drift_status = check_data_drift()

    assert isinstance(drift_status, bool)
    assert drift_status is True


def test_train_and_register_model_execution():
    """Verify model training and MLflow registration."""

    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    accuracy = train_and_register_model()

    assert accuracy >= 0.90

    experiment = mlflow.get_experiment_by_name(
        "iris-continuous-retraining"
    )

    assert experiment is not None

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id]
    )

    assert len(runs) > 0

    latest_run = runs.iloc[0]

    assert "metrics.accuracy" in runs.columns
    assert latest_run["metrics.accuracy"] >= 0.90

    registered_models = mlflow.search_registered_models()

    model_names = [
        registered_model.name
        for registered_model in registered_models
    ]

    assert MODEL_NAME in model_names


def test_full_retrain_pipeline_trigger(capsys):
    """Verify end-to-end retraining pipeline execution."""

    run_pipeline()

    captured = capsys.readouterr()

    assert "Checking production data for drift..." in captured.out

    assert (
        "Data drift confirmed! Triggering retraining pipeline..."
        in captured.out
    )

    assert "Initiating retraining process..." in captured.out
