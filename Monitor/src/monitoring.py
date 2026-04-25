from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

def generate_drift_report(reference_data, current_data):
    report = Report(metrics=[DataDriftPreset()])

    report.run(
        reference_data=reference_data,
        current_data=current_data
    )

    return report