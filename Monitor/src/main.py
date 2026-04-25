from data_loader import load_data
from monitoring import generate_drift_report

def main():
    # Step 1: Load data
    reference_data, current_data = load_data()

    # Step 2: Generate report
    report = generate_drift_report(reference_data, current_data)

    # Step 3: Save report
    report.save_html("reports/data_drift_report.html")

    print("✅ Report generated successfully!")

if __name__ == "__main__":
    main()