
from pathlib import Path

from scipy.stats import ks_2samp
from sklearn.datasets import load_iris

from evidently import Report
from evidently.presets import DataDriftPreset


# Project directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

REPORT_PATH = PROJECT_ROOT / "drift_report.html"


def check_data_drift():

    # Step 1: Load Iris dataset
    print("Loading Iris reference dataset...")

    iris = load_iris(as_frame=True)

    reference = iris.data.sample(
        n=100,
        random_state=42
    ).copy()

    # Step 2: Simulate production data
    current = reference.copy()

    current["sepal length (cm)"] *= 1.5
    current["sepal width (cm)"] *= 1.5

    print("Comparing reference and production data...")

    # Step 3: Statistical drift detection
    drifted_columns = []

    for column in reference.columns:

        statistic, p_value = ks_2samp(
            reference[column],
            current[column]
        )

        print(f"{column}: p-value = {p_value:.6f}")

        if p_value < 0.05:
            drifted_columns.append(column)

    dataset_drift = len(drifted_columns) > 0

    print(f"Drifted Features: {drifted_columns}")
    print(f"Dataset Drift Detected: {dataset_drift}")

    # Step 4: Generate Evidently report
    try:

        print("Generating Evidently drift report...")

        report = Report(
            metrics=[DataDriftPreset()]
        )

        snapshot = report.run(
            reference_data=reference,
            current_data=current
        )

        snapshot.save_html(str(REPORT_PATH))

        print(f"Drift report saved at: {REPORT_PATH}")

    except Exception as error:

        print(f"Warning: Evidently report generation failed: {error}")

    return dataset_drift


if __name__ == "__main__":

    drift_status = check_data_drift()

    print("Final Drift Status:", drift_status)
