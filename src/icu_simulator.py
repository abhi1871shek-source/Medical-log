"""
ICU Patient Vitals Simulator & Utility-Based Risk Scoring Agent
==============================================================
Part 1: Simulates 8 ICU patients with real-time physiological vitals:
  - Heart Rate (HR, bpm)
  - Blood Pressure: Systolic (SBP, mmHg) & Diastolic (DBP, mmHg)
  - Oxygen Saturation (SpO2, %)
  - Body Temperature (Temp, °C)
  - Respiratory Rate (RR, breaths/min)

AI Agent Type: Utility-Based Agent
  The agent calculates a continuous Utility/Risk metric U(s) over the patient's
  vital state vector. High clinical deviation from homeostatic norms decreases
  the patient's health utility and increases the deterioration risk score (0-100).
  When Risk Score >= 65, the agent triggers a HIGH RISK alert.
"""

import numpy as np
import pandas as pd


# -------------------------------------------------------------
# Utility Function: Clinical Risk Calculation (0 to 100)
# -------------------------------------------------------------
def calculate_vital_risk_score(hr: float, sbp: float, dbp: float, spo2: float, temp: float, rr: float) -> float:
    """
    Computes clinical risk score based on deviation from normal physiological ranges.
    Inspired by modified early warning score (MEWS) principles mapped to a 0-100 scale.

    Normal ranges:
      HR: 60 - 100 bpm
      SBP: 90 - 120 mmHg (DBP: 60 - 80 mmHg)
      SpO2: 95 - 100 %
      Temp: 36.5 - 37.5 °C
      RR: 12 - 20 breaths/min
    """
    risk = 0.0

    # 1. Heart Rate Risk (Max 25 pts)
    if hr < 40 or hr > 130:
        risk += 25.0
    elif hr < 50 or hr > 110:
        risk += 16.0
    elif hr < 60 or hr > 100:
        risk += 8.0

    # 2. Blood Pressure (Systolic) Risk (Max 25 pts)
    if sbp < 75 or sbp > 190:
        risk += 25.0
    elif sbp < 90 or sbp > 150:
        risk += 16.0
    elif sbp < 100 or sbp > 135:
        risk += 8.0

    # 3. Oxygen Saturation (SpO2) Risk (Max 25 pts)
    if spo2 < 85:
        risk += 25.0
    elif spo2 < 90:
        risk += 20.0
    elif spo2 < 93:
        risk += 14.0
    elif spo2 < 95:
        risk += 6.0

    # 4. Respiratory Rate Risk (Max 15 pts)
    if rr < 8 or rr > 30:
        risk += 15.0
    elif rr < 10 or rr > 24:
        risk += 10.0
    elif rr < 12 or rr > 20:
        risk += 5.0

    # 5. Temperature Risk (Max 10 pts)
    if temp < 35.0 or temp > 39.2:
        risk += 10.0
    elif temp < 36.0 or temp > 38.3:
        risk += 6.0
    elif temp < 36.5 or temp > 37.8:
        risk += 3.0

    # Clamp risk between 0 and 100
    return float(np.clip(risk, 0.0, 100.0))


def get_risk_category(risk_score: float) -> str:
    """Categorizes risk score into clinical status."""
    if risk_score >= 65.0:
        return "HIGH RISK"
    elif risk_score >= 35.0:
        return "MODERATE RISK"
    else:
        return "NORMAL / LOW RISK"


# -------------------------------------------------------------
# 8 ICU Patients Profiles & Profiles Metadata
# -------------------------------------------------------------
PATIENT_PROFILES = [
    {
        "bed_id": 1,
        "patient_id": "PAT-101",
        "name": "Sarah Connor",
        "age": 45,
        "diagnosis": "Post-Op Appendectomy",
        "trend": "stable",
        "deteriorates": False,
        "base_hr": 74, "base_sbp": 118, "base_dbp": 76, "base_spo2": 98, "base_temp": 36.8, "base_rr": 15
    },
    {
        "bed_id": 2,
        "patient_id": "PAT-102",
        "name": "Robert Miller",
        "age": 68,
        "diagnosis": "Mild COPD & Bronchitis",
        "trend": "moderate_fluctuating",
        "deteriorates": False,
        "base_hr": 88, "base_sbp": 128, "base_dbp": 82, "base_spo2": 93, "base_temp": 37.3, "base_rr": 21
    },
    {
        "bed_id": 3,
        "patient_id": "PAT-103",
        "name": "Elena Vasquez",
        "age": 58,
        "diagnosis": "Severe Urosepsis (Deteriorating)",
        "trend": "septic_shock",
        "deteriorates": True,
        "deteriorate_start_hour": 10,
        "base_hr": 82, "base_sbp": 116, "base_dbp": 74, "base_spo2": 97, "base_temp": 37.2, "base_rr": 16
    },
    {
        "bed_id": 4,
        "patient_id": "PAT-104",
        "name": "David Kim",
        "age": 62,
        "diagnosis": "Acute Myocardial Infarction",
        "trend": "cardiac_decompensation",
        "deteriorates": True,
        "deteriorate_start_hour": 15,
        "base_hr": 78, "base_sbp": 130, "base_dbp": 85, "base_spo2": 96, "base_temp": 36.9, "base_rr": 17
    },
    {
        "bed_id": 5,
        "patient_id": "PAT-105",
        "name": "Amina Yusuf",
        "age": 52,
        "diagnosis": "Ischemic Stroke (Post-TPA)",
        "trend": "stable",
        "deteriorates": False,
        "base_hr": 72, "base_sbp": 122, "base_dbp": 78, "base_spo2": 98, "base_temp": 36.7, "base_rr": 14
    },
    {
        "bed_id": 6,
        "patient_id": "PAT-106",
        "name": "James Henderson",
        "age": 39,
        "diagnosis": "Diabetic Ketoacidosis (Recovering)",
        "trend": "improving",
        "deteriorates": False,
        "base_hr": 92, "base_sbp": 112, "base_dbp": 70, "base_spo2": 97, "base_temp": 37.1, "base_rr": 22
    },
    {
        "bed_id": 7,
        "patient_id": "PAT-107",
        "name": "Maria Gonzalez",
        "age": 71,
        "diagnosis": "Acute Respiratory Distress (ARDS)",
        "trend": "respiratory_failure",
        "deteriorates": True,
        "deteriorate_start_hour": 7,
        "base_hr": 85, "base_sbp": 124, "base_dbp": 80, "base_spo2": 94, "base_temp": 37.4, "base_rr": 23
    },
    {
        "bed_id": 8,
        "patient_id": "PAT-108",
        "name": "Thomas Wright",
        "age": 29,
        "diagnosis": "Blunt Trauma (Observation)",
        "trend": "stable",
        "deteriorates": False,
        "base_hr": 76, "base_sbp": 120, "base_dbp": 78, "base_spo2": 99, "base_temp": 36.6, "base_rr": 15
    }
]


def simulate_patient_trajectory(profile: dict, total_hours: int = 24, seed: int = 42) -> pd.DataFrame:
    """
    Simulates vital signs for a single ICU patient over `total_hours` (1 to 24).
    Applies physiological drift, noise, and progression for deteriorating cases.
    """
    rng = np.random.RandomState(seed + profile["bed_id"] * 101)
    records = []

    hr = profile["base_hr"]
    sbp = profile["base_sbp"]
    dbp = profile["base_dbp"]
    spo2 = profile["base_spo2"]
    temp = profile["base_temp"]
    rr = profile["base_rr"]

    deteriorates = profile.get("deteriorates", False)
    det_start = profile.get("deteriorate_start_hour", 12)
    trend = profile.get("trend", "stable")

    for hour in range(1, total_hours + 1):
        # Base random walk noise
        hr += rng.normal(0, 1.5)
        sbp += rng.normal(0, 1.8)
        dbp += rng.normal(0, 1.2)
        spo2 += rng.normal(0, 0.4)
        temp += rng.normal(0, 0.05)
        rr += rng.normal(0, 0.6)

        # Deterioration mechanics
        if deteriorates and hour >= det_start:
            progress = (hour - det_start + 1)
            if trend == "septic_shock":
                # Severe Sepsis: Tachycardia, Hypotension, High Fever, Tachypnea, hypoxia
                hr += 2.8 * progress * 0.4
                sbp -= 2.6 * progress * 0.4
                dbp -= 1.8 * progress * 0.35
                temp += 0.12 * progress * 0.35
                rr += 0.8 * progress * 0.35
                spo2 -= 0.6 * progress * 0.35
            elif trend == "cardiac_decompensation":
                # AMI / Cardiogenic shock: severe tachycardia/arrhythmia, severe hypotension
                hr += (rng.choice([-1, 1]) * 4.0) + 1.8 * progress * 0.4
                sbp -= 3.2 * progress * 0.4
                dbp -= 1.9 * progress * 0.35
                rr += 0.7 * progress * 0.3
                spo2 -= 0.5 * progress * 0.3
            elif trend == "respiratory_failure":
                # ARDS: plunging SpO2, severe tachypnea, compensatory tachycardia
                spo2 -= 1.1 * progress * 0.4
                rr += 1.0 * progress * 0.4
                hr += 2.0 * progress * 0.35
                sbp -= 1.2 * progress * 0.25

        elif trend == "improving" and hour > 6:
            # Gradually moves closer to optimal
            hr += (78 - hr) * 0.08
            sbp += (118 - sbp) * 0.08
            rr += (16 - rr) * 0.08

        # Clip values to plausible human physiological boundaries
        hr_c = round(float(np.clip(hr, 30, 210)), 1)
        sbp_c = round(float(np.clip(sbp, 50, 240)), 1)
        dbp_c = round(float(np.clip(dbp, 30, 140)), 1)
        spo2_c = round(float(np.clip(spo2, 70, 100)), 1)
        temp_c = round(float(np.clip(temp, 34.0, 42.0)), 2)
        rr_c = round(float(np.clip(rr, 6, 45)), 1)

        risk_score = calculate_vital_risk_score(hr_c, sbp_c, dbp_c, spo2_c, temp_c, rr_c)
        risk_category = get_risk_category(risk_score)
        is_high_risk = 1 if risk_score >= 65.0 else 0

        records.append({
            "hour": hour,
            "bed_id": profile["bed_id"],
            "patient_id": profile["patient_id"],
            "name": profile["name"],
            "age": profile["age"],
            "diagnosis": profile["diagnosis"],
            "heart_rate": hr_c,
            "systolic_bp": sbp_c,
            "diastolic_bp": dbp_c,
            "spo2": spo2_c,
            "temperature": temp_c,
            "respiratory_rate": rr_c,
            "risk_score": round(risk_score, 1),
            "risk_category": risk_category,
            "is_high_risk": is_high_risk
        })

    return pd.DataFrame(records)


def simulate_all_icu_beds(total_hours: int = 24, seed: int = 42) -> pd.DataFrame:
    """Simulates 24-hour trajectories for all 8 ICU patients."""
    dfs = []
    for profile in PATIENT_PROFILES:
        df_p = simulate_patient_trajectory(profile, total_hours=total_hours, seed=seed)
        dfs.append(df_p)
    return pd.concat(dfs, ignore_index=True)


def generate_training_dataset(num_patients: int = 120, hours_per_patient: int = 24, seed: int = 999) -> pd.DataFrame:
    """
    Generates a synthetic ICU cohort (~2,800 rows) with varied patient phenotypes
    for robust machine learning training and evaluation.
    """
    rng = np.random.RandomState(seed)
    all_data = []

    for p in range(1, num_patients + 1):
        p_id = f"SYNTH-{p:04d}"
        # Decide if patient deteriorates (approx 30% positive class prevalence)
        det = rng.rand() < 0.30
        trend_type = rng.choice(["septic_shock", "cardiac_decompensation", "respiratory_failure"]) if det else rng.choice(["stable", "moderate_fluctuating", "improving"])
        det_start = rng.randint(4, 16) if det else 24

        base_profile = {
            "bed_id": (p % 8) + 1,
            "patient_id": p_id,
            "name": f"Synthetic Patient {p}",
            "age": int(rng.randint(25, 85)),
            "diagnosis": f"Condition {trend_type}",
            "trend": trend_type,
            "deteriorates": det,
            "deteriorate_start_hour": det_start,
            "base_hr": float(rng.normal(80, 10)),
            "base_sbp": float(rng.normal(120, 12)),
            "base_dbp": float(rng.normal(78, 8)),
            "base_spo2": float(rng.normal(97, 1.5)),
            "base_temp": float(rng.normal(37.0, 0.4)),
            "base_rr": float(rng.normal(16, 2.5))
        }

        df_single = simulate_patient_trajectory(base_profile, total_hours=hours_per_patient, seed=seed + p)
        all_data.append(df_single)

    return pd.concat(all_data, ignore_index=True)
