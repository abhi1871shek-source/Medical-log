"""
AI-Powered Medical Log Allocation & ICU Predictive Alert System
==============================================================
Part 3: Streamlit Interactive Dashboard
Features:
  - Real-time ICU 8-Bed Status Board with Color-Coded Risk (Green/Yellow/Red)
  - Time Simulation Slider (Hour 1 to 24) demonstrating patient deterioration
  - Live Alert Feed with automated clinical warnings
  - Patient Vitals Trajectory Chart with marked Alert Threshold & Trigger Point
  - Scikit-Learn Model Diagnostics (Accuracy, Confusion Matrix, False Alarm Rate)
  - Goal-Based Medical Record Allocator & Full-Text / Multi-Filter Search
  - Audit Allocation Log and Document Viewer
"""

import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# Ensure src modules can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.icu_simulator import (
    simulate_all_icu_beds,
    PATIENT_PROFILES,
    calculate_vital_risk_score,
    get_risk_category
)
from src.icu_model import (
    ICUAlertPredictor,
    plot_patient_vitals_alert,
    FEATURE_COLS
)
from src.sample_records import create_sample_record_files
from src.log_allocator import MedicalLogAllocatorAgent


# -------------------------------------------------------------
# Streamlit Page Configuration & Styling
# -------------------------------------------------------------
st.set_page_config(
    page_title="AI Medical Alert & Log Allocation System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Telemetry Dashboard Styling
st.markdown("""
<style>
    /* 1. Header transparency */
    header[data-testid="stHeader"],
    .stAppHeader {
        background: transparent !important;
        background-color: transparent !important;
    }
    
    /* 2. Global Background and Text */
    .stApp {
        background: #0B1120;
        color: #F9FAFB;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    /* 3. Metric cards (st.metric): Labels light grey (#9CA3AF), values bright white (#FFFFFF) */
    div[data-testid="stMetric"] {
        background-color: #1E293B !important;
        border: 1px solid #334155 !important;
        border-radius: 12px !important;
        padding: 16px 20px !important;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3) !important;
    }
    
    div[data-testid="stMetricLabel"],
    div[data-testid="stMetricLabel"] p,
    div[data-testid="stMetricLabel"] span,
    div[data-testid="stMetricLabel"] label {
        color: #9CA3AF !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px !important;
        text-transform: uppercase !important;
    }
    
    div[data-testid="stMetricValue"],
    div[data-testid="stMetricValue"] div,
    div[data-testid="stMetricValue"] span {
        color: #FFFFFF !important;
        font-size: 2.1rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.5px !important;
        line-height: 1.2 !important;
    }
    
    div[data-testid="stMetricDelta"],
    div[data-testid="stMetricDelta"] span {
        font-size: 0.82rem !important;
        font-weight: 600 !important;
    }

    /* 4. Tab Labels: Inactive light grey (#9CA3AF), Active bright cyan (#38BDF8) with underline */
    div[data-testid="stTabs"] button,
    button[data-baseweb="tab"] {
        color: #9CA3AF !important;
        font-size: 1.0rem !important;
        font-weight: 600 !important;
        padding: 12px 22px !important;
        background: transparent !important;
        border-bottom: 2px solid transparent !important;
        transition: color 0.15s ease-in-out, border 0.15s ease-in-out !important;
    }
    
    div[data-testid="stTabs"] button:hover,
    button[data-baseweb="tab"]:hover {
        color: #F3F4F6 !important;
    }
    
    div[data-testid="stTabs"] button[aria-selected="true"],
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #38BDF8 !important;
        font-weight: 700 !important;
        border-bottom: 3px solid #38BDF8 !important;
    }
    
    div[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
        background-color: #38BDF8 !important;
    }

    /* 5. Sidebar Styling & Contrast */
    section[data-testid="stSidebar"] {
        background-color: #0F172A !important;
        border-right: 1px solid #1E293B !important;
    }
    
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label {
        color: #F9FAFB !important;
    }
    
    section[data-testid="stSidebar"] .stCaption,
    section[data-testid="stSidebar"] small {
        color: #9CA3AF !important;
    }

    /* 6. Header Card */
    .main-header {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 22px 28px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px -2px rgba(0,0,0,0.5);
    }
    
    .main-header h1 {
        font-size: 1.85rem;
        font-weight: 800;
        color: #FFFFFF;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .main-header p {
        color: #CBD5E1;
        font-size: 0.95rem;
        margin: 6px 0 0 0;
    }

    /* 7. Bed Card Styling: High Contrast Borders & Badges */
    .bed-card {
        background: #1E293B;
        border-radius: 12px;
        padding: 18px;
        border: 1px solid #334155;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
        margin-bottom: 12px;
    }
    
    .bed-card.normal {
        border-left: 6px solid #10B981;
        background: linear-gradient(180deg, rgba(16, 185, 129, 0.10) 0%, #1E293B 100%);
    }
    
    .bed-card.moderate {
        border-left: 6px solid #F59E0B;
        background: linear-gradient(180deg, rgba(245, 158, 11, 0.10) 0%, #1E293B 100%);
    }
    
    .bed-card.high {
        border-left: 6px solid #EF4444;
        background: linear-gradient(180deg, rgba(239, 68, 68, 0.28) 0%, #1E293B 100%);
        box-shadow: 0 0 16px rgba(239, 68, 68, 0.35);
    }
    
    .badge {
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        display: inline-block;
        text-transform: uppercase;
    }
    
    .badge-normal {
        background-color: rgba(16, 185, 129, 0.25);
        color: #34D399;
        border: 1px solid #10B981;
    }
    
    .badge-moderate {
        background-color: rgba(245, 158, 11, 0.25);
        color: #FBBF24;
        border: 1px solid #F59E0B;
    }
    
    .badge-high {
        background-color: rgba(239, 68, 68, 0.35);
        color: #FCA5A5;
        border: 1px solid #EF4444;
    }

    /* Metric pill inside Bed Cards */
    .metric-pill {
        background: #0B1120;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 8px 10px;
        text-align: center;
    }
    .metric-pill .label {
        font-size: 0.68rem;
        color: #9CA3AF;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.4px;
    }
    .metric-pill .val {
        font-size: 1.05rem;
        font-weight: 800;
        color: #FFFFFF;
    }

    /* 8. Clear Red Alert Banner when beds in distress */
    .alert-banner {
        background: linear-gradient(90deg, #450A0A 0%, #250606 100%);
        border: 2px solid #EF4444;
        border-left: 8px solid #EF4444;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 14px;
        color: #FFFFFF;
        box-shadow: 0 4px 16px rgba(239, 68, 68, 0.35);
    }
    
    /* Expanders & Code blocks */
    div[data-testid="stExpander"] {
        background-color: #1E293B !important;
        border: 1px solid #334155 !important;
        border-radius: 10px !important;
    }
    
    div[data-testid="stExpander"] details summary span {
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------------------
# Cached Data & Model Initialization
# -------------------------------------------------------------
@st.cache_data
def get_simulation_data():
    """Generates 24-hour simulation data for the 8 ICU beds."""
    return simulate_all_icu_beds(total_hours=24, seed=42)


@st.cache_resource
def get_trained_model():
    """Trains and caches the Random Forest model and performance metrics."""
    predictor = ICUAlertPredictor(random_state=42)
    metrics = predictor.train_and_evaluate()
    return predictor, metrics


@st.cache_resource
def get_allocated_records():
    """Ensures sample records exist, runs Goal-Based Agent, and returns catalog & agent."""
    unallocated_dir = "data/unallocated_records"
    organized_dir = "data/organized_records"

    # Create unallocated records if not already present
    if not os.path.exists(unallocated_dir) or len(os.listdir(unallocated_dir)) < 20:
        create_sample_record_files(unallocated_dir)

    agent = MedicalLogAllocatorAgent(unallocated_dir, organized_dir)
    catalog = agent.allocate_all_records()
    return agent, catalog


# Load cached data and services
beds_df = get_simulation_data()
predictor, model_metrics = get_trained_model()
allocator_agent, record_catalog = get_allocated_records()


# -------------------------------------------------------------
# Sidebar: Controls & AI Agent Overview
# -------------------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2869/2869684.png", width=64)
    st.title("ICU Central Control")
    st.caption("AI Agent Framework Demo")

    st.markdown("---")
    st.subheader("⏱️ Live Timeline Simulation")
    selected_hour = st.slider(
        "Current Simulation Hour (1 - 24):",
        min_value=1,
        max_value=24,
        value=20,
        help="Slide across time to see patient vitals evolve, risk scores recalculate, and alerts fire dynamically."
    )

    st.markdown("---")
    st.subheader("🤖 AI Agent Types")
    st.markdown(r"""
    **1. Utility-Based Agent** (ICU Alert):
    - Computes a multi-attribute utility / risk score $U(s) \in [0, 100]$ based on vital sign deviations.
    - Triggers clinical alerts when risk utility exceeds threshold.

    **2. Goal-Based Agent** (Log Allocation):
    - Goal: Classify every unstructured record by Patient, Dept & Type, file into folders, and index for fast search.
    """)

    st.markdown("---")
    st.caption("Built for AI Assignment • Simulated Data Only")


# -------------------------------------------------------------
# Main Application Header
# -------------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>🏥 AI-Powered Medical Log Allocation & ICU Predictive Alert System</h1>
    <p>Integrated Dual-Agent Clinical Architecture: Real-time Utility Risk Telemetry & Automated Goal-Driven Record Ingestion</p>
</div>
""", unsafe_allow_html=True)


# Quick KPI Bar
kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

current_hour_df = beds_df[beds_df["hour"] == selected_hour]
num_high_risk = len(current_hour_df[current_hour_df["risk_category"] == "HIGH RISK"])
num_mod_risk = len(current_hour_df[current_hour_df["risk_category"] == "MODERATE RISK"])
num_norm_risk = len(current_hour_df[current_hour_df["risk_category"] == "NORMAL / LOW RISK"])

with kpi_col1:
    st.metric("Total ICU Beds", "8 Beds", f"Active Hour: {selected_hour}:00")
with kpi_col2:
    st.metric("Critical Alerts (High Risk)", f"{num_high_risk} Patients", delta=f"{num_high_risk} Bed(s) in Distress", delta_color="inverse")
with kpi_col3:
    st.metric("RF Model Accuracy", f"{model_metrics['accuracy'] * 100:.1f}%", f"False Alarm: {model_metrics['false_alarm_rate'] * 100:.2f}%")
with kpi_col4:
    st.metric("Medical Records Allocated", f"{len(record_catalog)} Files", "Goal-Based Agent Active")


# -------------------------------------------------------------
# Tabs Interface
# -------------------------------------------------------------
tab_icu, tab_records, tab_model = st.tabs([
    "🚨 Part 1: Live ICU Predictive Alert System",
    "📂 Part 2: Medical Log Allocation & Search",
    "📊 Model Performance & Confusion Matrix"
])


# =============================================================
# TAB 1: ICU PREDICTIVE ALERT SYSTEM (Utility-Based Agent)
# =============================================================
with tab_icu:
    st.subheader(f"🛏️ Live ICU Bed Monitoring Board (Hour {selected_hour:02d}:00)")

    # 1. Alert Banner Section
    current_alerts = predictor.scan_icu_beds_for_alerts(beds_df, current_hour=selected_hour)

    if current_alerts:
        st.markdown("#### ⚠️ Active Clinical Alerts")
        for a in current_alerts:
            st.markdown(f"""
            <div class="alert-banner">
                <div>
                    <strong>🚨 ALERT: Bed {a['bed_id']} – HIGH RISK</strong> &nbsp;|&nbsp;
                    <strong>Patient:</strong> {a['patient_name']} ({a['patient_id']}) &nbsp;|&nbsp;
                    <strong>Diagnosis:</strong> {a['diagnosis']} &nbsp;|&nbsp;
                    <strong>Risk Score:</strong> <span style="font-weight:bold; color:#F87171;">{a['risk_score']:.1f}/100</span>
                    <br>
                    <small>Vitals: HR {a['vitals']['HR']} bpm &nbsp;•&nbsp; BP {a['vitals']['BP']} mmHg &nbsp;•&nbsp; SpO2 {a['vitals']['SpO2']} &nbsp;•&nbsp; Temp {a['vitals']['Temp']} &nbsp;•&nbsp; RR {a['vitals']['RR']}</small>
                </div>
                <div>
                    <span class="badge badge-high">AI Confidence: {a['pred_risk_probability']}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("")
    else:
        st.success(f"✓ All 8 ICU beds are stable at Hour {selected_hour}:00. No critical deterioration alerts triggered.")

    # 2. 8-Bed Live Grid (2 rows of 4 columns)
    st.markdown("#### 📋 8 ICU Bed Telemetry Status")
    row1_cols = st.columns(4)
    row2_cols = st.columns(4)

    for i in range(8):
        bed_id = i + 1
        bed_row = current_hour_df[current_hour_df["bed_id"] == bed_id].iloc[0]
        col = row1_cols[i] if i < 4 else row2_cols[i - 4]

        risk_cat = bed_row["risk_category"]
        if risk_cat == "HIGH RISK":
            card_class = "bed-card high"
            badge_class = "badge badge-high"
        elif risk_cat == "MODERATE RISK":
            card_class = "bed-card moderate"
            badge_class = "badge badge-moderate"
        else:
            card_class = "bed-card normal"
            badge_class = "badge badge-normal"

        with col:
            card_html = f"""<div class="{card_class}">
<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
<span style="font-size:1.1rem; font-weight:700; color:#F8FAFC;">Bed {bed_id}</span>
<span class="{badge_class}">{risk_cat}</span>
</div>
<div style="font-size:0.88rem; font-weight:600; color:#E2E8F0;">{bed_row['name']}</div>
<div style="font-size:0.75rem; color:#94A3B8; margin-bottom:12px;">{bed_row['diagnosis']}</div>
<div style="display:grid; grid-template-columns: 1fr 1fr; gap:6px; margin-bottom:12px;">
<div class="metric-pill">
<div class="label">Heart Rate</div>
<div class="val">{bed_row['heart_rate']:.0f} <span style="font-size:0.65rem;">bpm</span></div>
</div>
<div class="metric-pill">
<div class="label">Blood Pressure</div>
<div class="val">{bed_row['systolic_bp']:.0f}/{bed_row['diastolic_bp']:.0f}</div>
</div>
<div class="metric-pill">
<div class="label">SpO2</div>
<div class="val">{bed_row['spo2']:.0f}%</div>
</div>
<div class="metric-pill">
<div class="label">Resp. Rate</div>
<div class="val">{bed_row['respiratory_rate']:.0f} <span style="font-size:0.65rem;">/min</span></div>
</div>
</div>
<div>
<div style="display:flex; justify-content:space-between; font-size:0.75rem; color:#94A3B8; margin-bottom:3px;">
<span>Risk Utility Score</span>
<span style="font-weight:700; color:#F8FAFC;">{bed_row['risk_score']:.1f}/100</span>
</div>
</div>
</div>"""
            st.markdown(card_html, unsafe_allow_html=True)
            # Progress bar for risk
            risk_pct = min(1.0, max(0.0, bed_row["risk_score"] / 100.0))
            st.progress(risk_pct)

    # 3. Patient Vitals Deep Dive & Saved Matplotlib Plot
    st.markdown("---")
    st.subheader("📈 Patient Vitals Trajectory & Marked Alert Point")
    st.caption("Inspect 24-hour vital sign trends with threshold detection and alert marker.")

    col_select, col_chart = st.columns([1, 3])

    with col_select:
        selected_bed_id = st.selectbox(
            "Select Patient Bed to Inspect:",
            options=[p["bed_id"] for p in PATIENT_PROFILES],
            format_func=lambda b: f"Bed {b}: {PATIENT_PROFILES[b-1]['name']} ({'Deteriorating' if PATIENT_PROFILES[b-1]['deteriorates'] else 'Stable'})",
            index=2  # Bed 3 by default
        )
        patient_info = PATIENT_PROFILES[selected_bed_id - 1]
        st.markdown(f"""
        **Patient Profile:**
        - **Name:** {patient_info['name']}
        - **ID:** `{patient_info['patient_id']}`
        - **Age:** {patient_info['age']} yrs
        - **Diagnosis:** {patient_info['diagnosis']}
        - **Status:** {'⚠️ Deteriorating Trajectory' if patient_info['deteriorates'] else '✓ Stable Clinical Course'}
        """)

    with col_chart:
        # Plot patient chart
        patient_history = beds_df[beds_df["bed_id"] == selected_bed_id]
        temp_chart_path = f"outputs/vitals_bed_{selected_bed_id}.png"
        plot_patient_vitals_alert(patient_history, output_path=temp_chart_path, threshold_score=65.0)
        st.image(temp_chart_path, use_container_width=True)


# =============================================================
# TAB 2: MEDICAL LOG ALLOCATION (Goal-Based Agent)
# =============================================================
with tab_records:
    st.subheader("📂 Medical Log Allocation & Search (Goal-Based Agent)")
    st.markdown("""
    The **Goal-Based Agent** automatically inspects incoming clinical documents, identifies the patient,
    department, and record type via rule-based keyword matching, and files them into structured folders.
    """)

    # Search Bar & Filters
    col_search, col_f_pat, col_f_dept, col_f_type = st.columns([2, 1, 1, 1])

    with col_search:
        search_query = st.text_input("🔍 Full-Text Keyword Search:", placeholder="e.g. troponin, sepsis, ventilator, meropenem...")

    with col_f_pat:
        pat_options = ["All"] + sorted(list(set(r["patient_id"] for r in record_catalog)))
        selected_pat = st.selectbox("Filter Patient:", pat_options)

    with col_f_dept:
        dept_options = ["All"] + sorted(list(set(r["department"] for r in record_catalog)))
        selected_dept = st.selectbox("Filter Department:", dept_options)

    with col_f_type:
        type_options = ["All"] + sorted(list(set(r["record_type"] for r in record_catalog)))
        selected_type = st.selectbox("Filter Record Type:", type_options)

    # Perform search
    search_results = allocator_agent.search_records(
        query=search_query,
        patient_id=selected_pat,
        department=selected_dept,
        record_type=selected_type
    )

    st.markdown(f"**Found {len(search_results)} matching medical record(s)**")

    # Display Search Results
    if search_results:
        for rec in search_results:
            with st.expander(f"📄 {rec['record_id']} — {rec['department']} | {rec['record_type']} (Patient: {rec['patient_id']})"):
                col_meta1, col_meta2 = st.columns([1, 2])
                with col_meta1:
                    st.markdown(f"""
                    **Record Metadata:**
                    - **Record ID:** `{rec['record_id']}`
                    - **Patient ID:** `{rec['patient_id']}`
                    - **Department:** `{rec['department']}`
                    - **Record Type:** `{rec['record_type']}`
                    - **Destination Path:** `{rec['relative_path']}`
                    """)
                with col_meta2:
                    st.markdown("**Document Content:**")
                    st.code(rec["full_text"], language="text")
    else:
        st.warning("No records matched the selected query or filters.")

    # Allocation Audit Log Section
    st.markdown("---")
    st.subheader("📋 Agent Audit Allocation Logs")
    with st.expander("View Full Allocation Log (Printed during ingestion)", expanded=False):
        for log in allocator_agent.allocation_logs:
            st.text(log)


# =============================================================
# TAB 3: MODEL PERFORMANCE & CONFUSION MATRIX
# =============================================================
with tab_model:
    st.subheader("📊 Scikit-Learn Random Forest Classifier Diagnostics")

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("Test Accuracy", f"{model_metrics['accuracy'] * 100:.2f}%")
    with col_m2:
        st.metric("False Alarm Rate (FPR)", f"{model_metrics['false_alarm_rate'] * 100:.2f}%", help="False Positives / (False Positives + True Negatives)")
    with col_m3:
        st.metric("Sensitivity (Recall)", f"{model_metrics['sensitivity'] * 100:.2f}%")
    with col_m4:
        st.metric("Specificity", f"{model_metrics['specificity'] * 100:.2f}%")

    col_cm, col_imp = st.columns(2)

    with col_cm:
        st.markdown("#### 🎯 Confusion Matrix")
        cm = model_metrics["confusion_matrix"]
        cm_df = pd.DataFrame(
            cm,
            index=["Actual Normal (0)", "Actual High Risk (1)"],
            columns=["Predicted Normal (0)", "Predicted High Risk (1)"]
        )
        st.dataframe(cm_df, use_container_width=True)

        st.markdown(f"""
        <div style="display:grid; grid-template-columns: 1fr 1fr; gap:10px; margin-top:12px;">
            <div style="background:#0F172A; border:1px solid #10B981; border-radius:8px; padding:10px 14px;">
                <div style="color:#9CA3AF; font-size:0.75rem; font-weight:600; text-transform:uppercase;">True Negatives (TN)</div>
                <div style="color:#34D399; font-size:1.4rem; font-weight:800;">{model_metrics['tn']}</div>
                <div style="color:#CBD5E1; font-size:0.75rem;">Normal vitals correctly detected</div>
            </div>
            <div style="background:#0F172A; border:1px solid #EF4444; border-radius:8px; padding:10px 14px;">
                <div style="color:#9CA3AF; font-size:0.75rem; font-weight:600; text-transform:uppercase;">False Positives (FP)</div>
                <div style="color:#F87171; font-size:1.4rem; font-weight:800;">{model_metrics['fp']}</div>
                <div style="color:#CBD5E1; font-size:0.75rem;">False alarms (FPR: {model_metrics['false_alarm_rate'] * 100:.2f}%)</div>
            </div>
            <div style="background:#0F172A; border:1px solid #F59E0B; border-radius:8px; padding:10px 14px;">
                <div style="color:#9CA3AF; font-size:0.75rem; font-weight:600; text-transform:uppercase;">False Negatives (FN)</div>
                <div style="color:#FBBF24; font-size:1.4rem; font-weight:800;">{model_metrics['fn']}</div>
                <div style="color:#CBD5E1; font-size:0.75rem;">Missed high-risk conditions</div>
            </div>
            <div style="background:#0F172A; border:1px solid #10B981; border-radius:8px; padding:10px 14px;">
                <div style="color:#9CA3AF; font-size:0.75rem; font-weight:600; text-transform:uppercase;">True Positives (TP)</div>
                <div style="color:#34D399; font-size:1.4rem; font-weight:800;">{model_metrics['tp']}</div>
                <div style="color:#CBD5E1; font-size:0.75rem;">Critical deteriorations caught</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_imp:
        st.markdown("#### 🔍 Vitals Feature Importance")
        feat_df = pd.DataFrame(
            list(model_metrics["feature_importances"].items()),
            columns=["Vital Sign Feature", "Relative Importance"]
        ).sort_values("Relative Importance", ascending=True)

        fig, ax = plt.subplots(figsize=(6, 3.9))
        fig.patch.set_facecolor("#1E293B")
        ax.set_facecolor("#0F172A")
        bars = ax.barh(
            feat_df["Vital Sign Feature"],
            feat_df["Relative Importance"] * 100,
            color="#38BDF8",
            edgecolor="#0284C7",
            height=0.62
        )
        ax.bar_label(bars, fmt="%.1f%%", padding=6, color="#FFFFFF", fontweight="bold", fontsize=9.5)
        ax.set_xlim(0, max(feat_df["Relative Importance"] * 100) * 1.3)
        ax.tick_params(colors="#FFFFFF", labelsize=9.5)
        for spine in ax.spines.values():
            spine.set_color("#334155")
        ax.set_xlabel("Contribution (%)", color="#FFFFFF", fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.25, color="#475569")
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("💡 Artificial Intelligence Concept Comparison")
    col_agent1, col_agent2 = st.columns(2)
    with col_agent1:
        st.info("""
        **Part 1: Utility-Based Agent**
        - Operates with an internal multi-attribute **utility / penalty function** $U(S)$.
        - Rather than binary goals, it quantifies continuous physiological risk (0 to 100).
        - Balances multiple conflicting objectives: minimizing false alarms while guaranteeing early detection of clinical deterioration.
        """)
    with col_agent2:
        st.success("""
        **Part 2: Goal-Based Agent**
        - Operates with a specific **target goal state**: "All incoming medical records categorized into target directories with updated catalog".
        - Uses rule-based keyword matching condition-action rules to evaluate environmental inputs (text records).
        - Plans and executes actions (creates folders, writes files, logs actions) until the goal state is satisfied.
        """)
