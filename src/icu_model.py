"""
ICU Machine Learning Model & Alert Generator
============================================
Trains a scikit-learn Random Forest model on simulated physiological vitals.
Evaluates model performance:
  - Accuracy
  - Confusion Matrix
  - False Alarm Rate (False Positive Rate: FP / (FP + TN))
Generates real-time alerts for the 8 ICU beds.
Saves a matplotlib chart illustrating vitals progression and alert points.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from src.icu_simulator import (
    simulate_all_icu_beds,
    generate_training_dataset,
    calculate_vital_risk_score,
    get_risk_category
)

FEATURE_COLS = [
    "heart_rate",
    "systolic_bp",
    "diastolic_bp",
    "spo2",
    "temperature",
    "respiratory_rate"
]


class ICUAlertPredictor:
    """Utility-Based Predictive AI Agent for ICU Alert Monitoring."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            min_samples_split=5,
            random_state=self.random_state,
            class_weight="balanced"
        )
        self.is_trained = False
        self.metrics = {}

    def train_and_evaluate(self, train_df: pd.DataFrame = None):
        """Trains the Random Forest model and calculates evaluation metrics."""
        if train_df is None:
            train_df = generate_training_dataset(num_patients=100, hours_per_patient=24, seed=self.random_state)

        X = train_df[FEATURE_COLS]
        y = train_df["is_high_risk"]

        # Train / test split (80/20)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=self.random_state, stratify=y
        )

        self.model.fit(X_train, y_train)
        self.is_trained = True

        # Test set predictions
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        # Confusion Matrix breakdown: tn, fp, fn, tp
        tn, fp, fn, tp = cm.ravel()
        false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        self.metrics = {
            "accuracy": float(acc),
            "confusion_matrix": cm,
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "false_alarm_rate": float(false_alarm_rate),
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "feature_importances": dict(zip(FEATURE_COLS, self.model.feature_importances_))
        }

        return self.metrics

    def predict_vitals(self, vitals_dict: dict):
        """Predicts risk probability for a single vitals observation."""
        if not self.is_trained:
            raise ValueError("Model has not been trained yet. Call train_and_evaluate() first.")

        df = pd.DataFrame([vitals_dict])[FEATURE_COLS]
        pred_label = self.model.predict(df)[0]
        pred_prob = self.model.predict_proba(df)[0, 1]
        return int(pred_label), float(pred_prob)

    def scan_icu_beds_for_alerts(self, beds_df: pd.DataFrame, current_hour: int = None) -> list:
        """
        Scans all 8 ICU beds at the specified hour (or latest hour) and returns alerts.
        """
        if current_hour is None:
            current_hour = beds_df["hour"].max()

        latest_readings = beds_df[beds_df["hour"] == current_hour]
        alerts = []

        for _, row in latest_readings.iterrows():
            # Check model prediction and utility risk score
            pred_label, pred_prob = self.predict_vitals(row.to_dict())
            risk_score = row["risk_score"]

            # An alert is triggered if risk score >= 65 OR model prediction is positive
            if pred_label == 1 or risk_score >= 65.0:
                severity = "CRITICAL HIGH RISK" if risk_score >= 75 else "HIGH RISK"
                alerts.append({
                    "bed_id": int(row["bed_id"]),
                    "patient_id": row["patient_id"],
                    "patient_name": row["name"],
                    "diagnosis": row["diagnosis"],
                    "hour": int(row["hour"]),
                    "severity": severity,
                    "risk_score": float(risk_score),
                    "pred_risk_probability": round(pred_prob * 100, 1),
                    "vitals": {
                        "HR": row["heart_rate"],
                        "BP": f"{row['systolic_bp']:.0f}/{row['diastolic_bp']:.0f}",
                        "SpO2": f"{row['spo2']:.0f}%",
                        "Temp": f"{row['temperature']:.1f} C",
                        "RR": row["respiratory_rate"]
                    },
                    "message": f"ALERT: Bed {int(row['bed_id'])} - HIGH RISK (Patient: {row['name']} | Risk: {risk_score:.1f}/100)"
                })

        return alerts


# -------------------------------------------------------------
# Matplotlib Visualization: Patient Trajectory with Alert Point
# -------------------------------------------------------------
def plot_patient_vitals_alert(
    patient_df: pd.DataFrame,
    output_path: str = "outputs/vitals_alert_plot.png",
    threshold_score: float = 65.0
):
    """
    Plots a multi-panel chart of patient vitals over 24 hours.
    Marks the exact hour where deterioration crosses the HIGH RISK alert threshold.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    patient_name = patient_df["name"].iloc[0]
    bed_id = patient_df["bed_id"].iloc[0]
    diagnosis = patient_df["diagnosis"].iloc[0]

    # Find the first hour where risk crossed threshold
    alert_hours = patient_df[patient_df["risk_score"] >= threshold_score]["hour"].tolist()
    first_alert_hour = alert_hours[0] if alert_hours else None

    # Styling settings
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(4, 1, figsize=(11, 10.5), sharex=True)
    fig.patch.set_facecolor("#0B1120")  # Matches Streamlit main background

    hours = patient_df["hour"]

    # Common subplot styling helper
    def style_ax(ax, title, ylabel):
        ax.set_facecolor("#1E293B")  # Matches Streamlit card background
        ax.set_title(title, fontsize=12, fontweight="bold", color="#FFFFFF", pad=8, loc="left")
        ax.set_ylabel(ylabel, fontsize=10, color="#E2E8F0", fontweight="semibold")
        ax.tick_params(colors="#CBD5E1", labelsize=9)
        ax.grid(True, linestyle="--", alpha=0.35, color="#475569")
        for spine in ax.spines.values():
            spine.set_color("#334155")
        if first_alert_hour:
            ax.axvline(x=first_alert_hour, color="#EF4444", linestyle="--", linewidth=2.2, alpha=0.95)

    # 1. Heart Rate & Blood Pressure
    ax1 = axes[0]
    ax1.plot(hours, patient_df["heart_rate"], color="#38BDF8", linewidth=2.4, marker="o", markersize=4, label="Heart Rate (bpm)")
    ax1.plot(hours, patient_df["systolic_bp"], color="#F43F5E", linewidth=2.2, linestyle="-.", label="Systolic BP (mmHg)")
    ax1.plot(hours, patient_df["diastolic_bp"], color="#FB7185", linewidth=1.8, linestyle=":", label="Diastolic BP (mmHg)")
    # Safe reference zones
    ax1.axhspan(60, 100, color="#10B981", alpha=0.12, label="Normal HR (60-100)")
    style_ax(ax1, f"Bed {bed_id}: {patient_name} - Hemodynamics (HR & BP)", "bpm / mmHg")
    ax1.legend(loc="upper left", facecolor="#0B1120", edgecolor="#334155", labelcolor="#FFFFFF", fontsize=8.5)

    # 2. Oxygen Saturation (SpO2)
    ax2 = axes[1]
    ax2.plot(hours, patient_df["spo2"], color="#34D399", linewidth=2.5, marker="s", markersize=4, label="SpO2 (%)")
    ax2.axhspan(95, 100, color="#10B981", alpha=0.12, label="Normal SpO2 (>=95%)")
    ax2.axhline(y=90, color="#EF4444", linestyle=":", linewidth=1.8, label="Hypoxia Critical (<90%)")
    style_ax(ax2, "Oxygen Saturation (SpO2)", "SpO2 (%)")
    ax2.set_ylim(min(patient_df["spo2"].min() - 4, 82), 101)
    ax2.legend(loc="lower left", facecolor="#0B1120", edgecolor="#334155", labelcolor="#FFFFFF", fontsize=8.5)

    # 3. Respiratory Rate & Body Temperature
    ax3 = axes[2]
    ax3_twin = ax3.twinx()
    p1 = ax3.plot(hours, patient_df["respiratory_rate"], color="#FBBF24", linewidth=2.4, marker="^", markersize=4, label="Resp. Rate (bpm)")
    p2 = ax3_twin.plot(hours, patient_df["temperature"], color="#FB923C", linewidth=2.2, linestyle="--", label="Temp (C)")
    style_ax(ax3, "Respiratory Rate & Body Temperature", "RR (breaths/min)")
    ax3_twin.set_ylabel("Temp (C)", color="#FB923C", fontsize=10, fontweight="bold")
    ax3_twin.tick_params(colors="#FB923C")
    ax3_twin.grid(False)
    lines = p1 + p2
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc="upper left", facecolor="#0B1120", edgecolor="#334155", labelcolor="#FFFFFF", fontsize=8.5)

    # 4. Utility Risk Score & Alert Point
    ax4 = axes[3]
    ax4.plot(hours, patient_df["risk_score"], color="#E11D48", linewidth=2.8, marker="D", markersize=5, label="Utility Risk Score")
    ax4.axhline(y=threshold_score, color="#EF4444", linestyle="--", linewidth=2.0, label=f"Alert Threshold ({threshold_score})")
    ax4.axhspan(threshold_score, 100, color="#EF4444", alpha=0.22, label="HIGH RISK ZONE")
    ax4.axhspan(35, threshold_score, color="#F59E0B", alpha=0.14, label="MODERATE RISK ZONE")
    ax4.axhspan(0, 35, color="#10B981", alpha=0.10, label="NORMAL ZONE")
    style_ax(ax4, "Composite Utility Risk Score & Alert Trigger Point", "Risk Score (0-100)")
    ax4.set_ylim(0, 105)
    ax4.set_xlabel("Time (Observation Hours: 1 to 24)", fontsize=11, color="#F9FAFB", fontweight="bold")
    ax4.set_xticks(range(1, 25))
    ax4.legend(loc="upper left", facecolor="#0B1120", edgecolor="#334155", labelcolor="#FFFFFF", fontsize=8.5)

    # Annotate Alert Point
    if first_alert_hour:
        first_alert_score = patient_df[patient_df["hour"] == first_alert_hour]["risk_score"].iloc[0]
        ax4.annotate(
            f"ALERT TRIGGERED!\nHour {first_alert_hour} | Risk: {first_alert_score:.1f}",
            xy=(first_alert_hour, first_alert_score),
            xytext=(max(1, first_alert_hour - 5), first_alert_score - 18),
            arrowprops=dict(facecolor="#F43F5E", edgecolor="#FFFFFF", shrink=0.05, width=2.2, headwidth=8.5),
            bbox=dict(boxstyle="round,pad=0.5", fc="#DC2626", ec="#FFFFFF", lw=1.8),
            color="#FFFFFF",
            fontweight="bold",
            fontsize=9.5
        )

    plt.suptitle(
        f"ICU Predictive Alert System: Patient Vitals Trajectory\nBed {bed_id} - {patient_name} ({diagnosis})",
        fontsize=14,
        fontweight="bold",
        color="#FFFFFF",
        y=0.99
    )

    plt.tight_layout()
    fig.subplots_adjust(top=0.92)
    plt.savefig(output_path, dpi=160, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    return output_path
