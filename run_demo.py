"""
================================================================================
AI-Powered Medical Log Allocation & ICU Predictive Alert System
Complete Assignment Execution Demo Script
================================================================================

This script executes:
  Part 1: ICU Predictive Alert System (Utility-Based Agent)
    - Simulates 8 ICU patients over 24 hours with physiological vitals
    - Evaluates clinical risk scores (utility function)
    - Trains a scikit-learn Random Forest model on simulated data
    - Emits real-time alerts (e.g. "ALERT: Bed 3 – HIGH RISK")
    - Displays Accuracy, Confusion Matrix, and False Alarm Rate
    - Saves a matplotlib vitals trajectory chart with the alert point marked

  Part 2: Medical Log Allocation (Goal-Based Agent)
    - Creates 20 sample simulated medical records (labs, Rx, notes)
    - Auto-classifies each record by Patient ID, Department, and Record Type
    - Organizes and stores them into hierarchical department/type folders
    - Prints an audit allocation log for each record
    - Demonstrates multi-criteria search functionality
================================================================================
"""

import os
import sys
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.icu_simulator import simulate_all_icu_beds, PATIENT_PROFILES
from src.icu_model import ICUAlertPredictor, plot_patient_vitals_alert
from src.sample_records import create_sample_record_files
from src.log_allocator import MedicalLogAllocatorAgent


def print_banner(title: str):
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)


def run_part_1_icu_predictive_alert():
    print_banner("PART 1: ICU PREDICTIVE ALERT SYSTEM (Utility-Based Agent)")

    print("\n[Step 1.1] Simulating 8 ICU Patients with real-time vitals over 24 hours...")
    beds_df = simulate_all_icu_beds(total_hours=24, seed=42)
    print(f"-> Generated {len(beds_df)} vitals observation records across 8 ICU beds.")

    print("\n[Patient Roster & Diagnoses]")
    for p in PATIENT_PROFILES:
        det_text = f"Deteriorating at Hour {p.get('deteriorate_start_hour', 'N/A')}" if p['deteriorates'] else "Stable / Monitoring"
        print(f"  Bed {p['bed_id']}: {p['name']:16s} | ID: {p['patient_id']} | Status: {det_text:<30s} | {p['diagnosis']}")

    # Initialize and train machine learning agent
    print("\n[Step 1.2] Training Scikit-Learn Random Forest Classifier on Simulated Cohort...")
    predictor = ICUAlertPredictor(random_state=42)
    metrics = predictor.train_and_evaluate()

    print("\n" + "-" * 50)
    print("       MODEL EVALUATION PERFORMANCE METRICS")
    print("-" * 50)
    print(f"  Accuracy:              {metrics['accuracy'] * 100:.2f}%")
    print(f"  False Alarm Rate (FPR): {metrics['false_alarm_rate'] * 100:.2f}%  (FP / [FP + TN])")
    print(f"  Sensitivity (Recall):  {metrics['sensitivity'] * 100:.2f}%")
    print(f"  Specificity:           {metrics['specificity'] * 100:.2f}%")
    print("\n  Confusion Matrix [TN, FP; FN, TP]:")
    cm = metrics["confusion_matrix"]
    print(f"              Predicted 0 (Normal)   Predicted 1 (High Risk)")
    print(f"  Actual 0:   {cm[0, 0]:<22d} {cm[0, 1]:<20d}  (True Negatives: {cm[0, 0]}, False Positives: {cm[0, 1]})")
    print(f"  Actual 1:   {cm[1, 0]:<22d} {cm[1, 1]:<20d}  (False Negatives: {cm[1, 0]}, True Positives: {cm[1, 1]})")
    print("-" * 50)

    # Feature importances
    print("\n[Vitals Risk Feature Importance Ranking]:")
    for feat, imp in sorted(metrics["feature_importances"].items(), key=lambda x: x[1], reverse=True):
        print(f"  - {feat:18s}: {imp * 100:.1f}% contribution")

    # Scan for alerts at peak observation hour (Hour 20)
    eval_hour = 20
    print(f"\n[Step 1.3] Scanning ICU Beds for Active Alerts at Hour {eval_hour}...")
    alerts = predictor.scan_icu_beds_for_alerts(beds_df, current_hour=eval_hour)

    if alerts:
        print("\n" + "!" * 78)
        print("                           ACTIVE ICU ALERTS")
        print("!" * 78)
        for a in alerts:
            print(f"  >>> ALERT: Bed {a['bed_id']} - HIGH RISK <<<")
            print(f"      Patient:    {a['patient_name']} ({a['patient_id']})")
            print(f"      Diagnosis:  {a['diagnosis']}")
            print(f"      Risk Score: {a['risk_score']:.1f}/100  | AI Alert Confidence: {a['pred_risk_probability']}%")
            print(f"      Vitals:     HR: {a['vitals']['HR']} bpm | BP: {a['vitals']['BP']} mmHg | SpO2: {a['vitals']['SpO2']} | Temp: {a['vitals']['Temp']} | RR: {a['vitals']['RR']}")
            print("  " + "-" * 74)
    else:
        print("  All beds currently within safe physiological limits.")

    # Generate and save vitals trajectory plot for Bed 3 (Elena Vasquez, septic shock)
    print("\n[Step 1.4] Generating and saving patient vitals graph with marked alert point...")
    bed3_df = beds_df[beds_df["bed_id"] == 3]
    output_chart = "outputs/vitals_alert_plot.png"
    saved_path = plot_patient_vitals_alert(bed3_df, output_path=output_chart, threshold_score=65.0)
    print(f"-> Successfully saved alert chart to: {os.path.abspath(saved_path)}")

    return beds_df, predictor, metrics


def run_part_2_medical_log_allocation():
    print_banner("PART 2: MEDICAL LOG ALLOCATION (Goal-Based Agent)")

    print("\n[Step 2.1] Generating 20 sample raw medical records...")
    unallocated_dir = "data/unallocated_records"
    organized_dir = "data/organized_records"

    created_paths = create_sample_record_files(dest_dir=unallocated_dir)
    print(f"-> Created {len(created_paths)} incoming medical documents in '{unallocated_dir}'.")

    print("\n[Step 2.2] Goal-Based Agent Auto-Classifying & Organizing Medical Records...")
    allocator_agent = MedicalLogAllocatorAgent(
        unallocated_dir=unallocated_dir,
        organized_dir=organized_dir
    )

    print("\n" + "-" * 78)
    print("                    AUDIT ALLOCATION LOGS")
    print("-" * 78)
    catalog = allocator_agent.allocate_all_records()
    print("-" * 78)
    print(f"-> All {len(catalog)} records successfully categorized and filed into '{organized_dir}'.")

    # Step 2.3: Demonstrate Search Functionality
    print("\n[Step 2.3] Demonstrating Medical Records Search Function...")

    # Query 1: Search by Patient ID PAT-103
    print("\n--- Search Query 1: Filter by Patient ID 'PAT-103' ---")
    results_p103 = allocator_agent.search_records(patient_id="PAT-103")
    for r in results_p103:
        print(f"  * [{r['record_id']}] {r['department']} | {r['record_type']} | Path: {r['file_path']}")

    # Query 2: Search by Department 'Cardiology'
    print("\n--- Search Query 2: Filter by Department 'Cardiology' ---")
    results_cardio = allocator_agent.search_records(department="Cardiology")
    for r in results_cardio:
        print(f"  * [{r['record_id']}] Patient: {r['patient_id']} | Type: {r['record_type']} | Path: {r['file_path']}")

    # Query 3: Full-text search for clinical term 'troponin'
    print("\n--- Search Query 3: Full-Text Keyword Search for 'troponin' ---")
    results_trop = allocator_agent.search_records(query="troponin")
    for r in results_trop:
        print(f"  * [{r['record_id']}] Patient: {r['patient_id']} | Dept: {r['department']} | Match Preview: {r['content_preview'][:100]}...")

    return allocator_agent


def main():
    print("=" * 78)
    print("  AI ASSIGNMENT DEMO: HEALTHCARE MULTI-AGENT SYSTEM")
    print("  AI-Powered Medical Log Allocation & ICU Predictive Alert System")
    print("=" * 78)

    # Ensure stdout handles UTF-8 on Windows
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    beds_df, predictor, metrics = run_part_1_icu_predictive_alert()
    allocator_agent = run_part_2_medical_log_allocation()

    print_banner("EXECUTION SUMMARY")
    print("  [OK] Part 1 (Utility-Based Agent): Completed successfully!")
    print(f"    - Random Forest Accuracy: {metrics['accuracy'] * 100:.2f}%")
    print(f"    - False Alarm Rate:       {metrics['false_alarm_rate'] * 100:.2f}%")
    print("    - Alert chart saved:      outputs/vitals_alert_plot.png")
    print("  [OK] Part 2 (Goal-Based Agent):    Completed successfully!")
    print(f"    - 20 records organized into: data/organized_records/")
    print("    - Search index stored:       data/organized_records/allocation_index.json")
    print("\nTo launch the interactive Part 3 Streamlit Web Dashboard, run:")
    print("    streamlit run app.py")
    print("=" * 78)


if __name__ == "__main__":
    main()
