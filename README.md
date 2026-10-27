# AI-Powered Medical Log Allocation & ICU Predictive Alert System

A dual-agent healthcare artificial intelligence demonstration project designed for academic evaluation. Built in **Python**, utilizing **Scikit-Learn**, **Pandas**, **NumPy**, **Matplotlib**, and **Streamlit**.

> **Note on Data Privacy:** This system operates strictly with **100% simulated, synthetic physiological data and fake clinical notes**. No real patient data or protected health information (PHI) is used.

---

## 📖 Table of Contents
1. [AI Agent Architecture & Concepts](#-ai-agent-architecture--concepts)
2. [Project Architecture & Directory Structure](#-project-architecture--directory-structure)
3. [Quick Start & How to Run](#-quick-start--how-to-run)
4. [Part 1: ICU Predictive Alert System (Utility-Based Agent)](#-part-1-icu-predictive-alert-system-utility-based-agent)
5. [Part 2: Medical Log Allocation (Goal-Based Agent)](#-part-2-medical-log-allocation-goal-based-agent)
6. [Part 3: Interactive Streamlit Dashboard](#-part-3-interactive-streamlit-dashboard)
7. [Experimental Results & Verification](#-experimental-results--verification)

---

## 🤖 AI Agent Architecture & Concepts

This project showcases two fundamental types of Artificial Intelligence agents from classical AI theory (Russell & Norvig):

| Agent Component | Agent Type | Goal / Formulation | Key AI Mechanism |
| :--- | :--- | :--- | :--- |
| **Part 1: ICU Alert System** | **Utility-Based Agent** | Quantify physiological risk into a continuous utility score $[0, 100]$ to balance early detection against false alarms. | Multi-attribute utility risk function + Scikit-Learn Random Forest Classifier |
| **Part 2: Medical Log Allocator** | **Goal-Based Agent** | Satisfy the state goal: *"Every incoming unstructured clinical record is correctly categorized and stored in its target directory with an audit log."* | Rule-based keyword matching, regex entity extraction, and automated file-system taxonomy |

### 1. Why is the ICU Alert System a Utility-Based Agent?
A simple goal-based agent might just check *"Is the patient alive or dead?"* But in intensive care, doctors care about the **degree of well-being or risk**. A utility-based agent maps the patient's continuous multidimensional physiological state $S_t = [\text{HR}, \text{SBP}, \text{DBP}, \text{SpO}_2, \text{Temp}, \text{RR}]$ to a real-valued utility / risk score $U(S_t) \in [0, 100]$. When the estimated deterioration risk crosses the critical safety threshold ($\ge 65.0$), the agent triggers an alert.

### 2. Why is the Medical Log Allocator a Goal-Based Agent?
A goal-based agent acts to achieve a specific target condition. In our medical records manager, the goal is:
$$\text{State}_{\text{unallocated}} \xrightarrow{\text{classify \& route}} \text{State}_{\text{organized \& indexed}}$$
The agent analyzes unstructured document contents, extracts identifying tokens, matches semantic rules to identify department and document type, generates appropriate folders, and updates the searchable catalog until all items achieve the goal state.

---

## 📁 Project Architecture & Directory Structure

```text
ai ass/
├── app.py                          # Part 3: Streamlit interactive web dashboard
├── run_demo.py                     # Unified CLI script executing Part 1 & Part 2 end-to-end
├── requirements.txt                # Python dependencies
├── README.md                       # Comprehensive documentation
│
├── src/
│   ├── __init__.py
│   ├── icu_simulator.py            # Patient vitals simulation (8 beds) & utility risk function
│   ├── icu_model.py                # Random Forest ML model, metrics & Matplotlib plot generator
│   ├── sample_records.py           # 20 realistic simulated medical documents generator
│   └── log_allocator.py            # Goal-based agent, keyword classifier & multi-filter search
│
├── data/
│   ├── unallocated_records/        # 20 raw incoming simulated medical records (.txt)
│   └── organized_records/          # Hierarchical folder structure by Department and Record Type:
│       ├── Cardiology/             #   [Doctor Note, Lab Report, Prescription]
│       ├── Pulmonology/            #   [Doctor Note, Lab Report, Prescription]
│       ├── Nephrology/             #   [Doctor Note, Lab Report]
│       ├── Neurology/              #   [Doctor Note, Prescription]
│       ├── Infectious Disease/     #   [Lab Report, Prescription]
│       ├── General Surgery/        #   [Discharge Summary, Doctor Note, Prescription]
│       ├── Trauma ICU/             #   [Doctor Note]
│       └── allocation_index.json   # Machine-readable indexed catalog of all records
│
└── outputs/
    └── vitals_alert_plot.png       # Matplotlib 4-panel trajectory graph with marked alert point
```

---

## 🚀 Quick Start & How to Run

### Step 1: Install Requirements
Open your terminal or command prompt in the project root directory and run:
```bash
pip install -r requirements.txt
```

### Step 2: Run the Terminal CLI Demo (Parts 1 & 2)
To run the automated agent pipeline, train the model, evaluate metrics, print active alerts, allocate records, and save the chart:
```bash
python run_demo.py
```

### Step 3: Launch the Interactive Streamlit Web Dashboard (Part 3)
To run the telemetry command center and record browser:
```bash
streamlit run app.py
```
Then open your browser at `http://localhost:8501`.

---

## 🚨 Part 1: ICU Predictive Alert System (Utility-Based Agent)

### 1. Simulated Patients & Vitals
The system monitors **8 active ICU beds** over a 24-hour observation timeline:
- **Bed 1 (Sarah Connor):** Post-Op Appendectomy — Stable vitals.
- **Bed 2 (Robert Miller):** Chronic COPD & Bronchitis — Moderate respiratory fluctuations.
- **Bed 3 (Elena Vasquez):** Severe Urosepsis — **Deteriorating** (Tachycardia, severe hypotension, high fever, elevated lactate, triggers high-risk alert).
- **Bed 4 (David Kim):** Acute Myocardial Infarction — **Deteriorating** (Unstable blood pressure, arrhythmia, elevated troponin).
- **Bed 5 (Amina Yusuf):** Ischemic Stroke (Post-TPA) — Stable neuro ICU observation.
- **Bed 6 (James Henderson):** Diabetic Ketoacidosis & AKI — Gradually recovering on insulin and IV hydration.
- **Bed 7 (Maria Gonzalez):** Acute Respiratory Distress (ARDS) — **Deteriorating** (Hypoxemia $\text{SpO}_2 < 85\%$, tachypnea, early high-risk alert).
- **Bed 8 (Thomas Wright):** Blunt Thoracic Trauma — Stable post-trauma recovery.

### 2. Utility-Based Risk Scoring Formula
The risk function $U(S) \in [0, 100]$ calculates weighted penalties across 5 vital sign channels based on clinical early warning score principles:
- **Heart Rate (HR):** Max 25 points penalty for bradycardia (< 40 bpm) or severe tachycardia (> 130 bpm).
- **Systolic BP (SBP):** Max 25 points penalty for hypotension (< 75 mmHg) or hypertensive crisis (> 190 mmHg).
- **Oxygen Saturation ($\text{SpO}_2$):** Max 25 points penalty for hypoxia (< 90%).
- **Respiratory Rate (RR):** Max 15 points penalty for tachypnea (> 30 bpm) or bradypnea (< 8 bpm).
- **Temperature (Temp):** Max 10 points penalty for hypothermia (< 35.0°C) or high fever (> 39.2°C).

**Risk Stratification:**
- **Normal / Low Risk:** $0 - 34$ (Green)
- **Moderate Risk:** $35 - 64$ (Yellow / Amber)
- **High Risk:** $\ge 65$ (Red — Alert Triggered)

### 3. Machine Learning Model & Evaluation
A Scikit-Learn **Random Forest Classifier** (`n_estimators=100`, `max_depth=6`) is trained on a simulated cohort of over 2,400 observations:
- **Accuracy:** `99.17%`
- **False Alarm Rate (False Positive Rate):** `0.44%` ($\frac{\text{FP}}{\text{FP} + \text{TN}} = \frac{2}{2 + 448}$)
- **Sensitivity (Recall):** `93.33%`
- **Specificity:** `99.56%`

#### Confusion Matrix:
| | Predicted Normal (0) | Predicted High Risk (1) |
| :--- | :---: | :---: |
| **Actual Normal (0)** | **448** (True Negatives) | **2** (False Positives / False Alarms) |
| **Actual High Risk (1)** | **2** (False Negatives) | **28** (True Positives) |

### 4. Saved Matplotlib Vitals Trajectory Chart
The script automatically exports `outputs/vitals_alert_plot.png` featuring 4 synchronized panels:
1. **Hemodynamics:** Heart Rate & Blood Pressure with normal range shading.
2. **Oxygenation:** $\text{SpO}_2$ with hypoxia critical line (< 90%).
3. **Respiration & Temperature:** Dual-axis plot of RR and Temperature.
4. **Utility Risk Score:** Risk progression across Normal, Moderate, and High Risk zones with an annotated arrow marking the exact alert trigger point.

---

## 📂 Part 2: Medical Log Allocation (Goal-Based Agent)

### 1. Simulated Clinical Records
The system generates **20 simulated medical records** spanning:
- **Lab Reports:** Complete Blood Counts, Arterial Blood Gases (ABG), Renal Panels, Cardiac Troponin & Echocardiography, Sputum Cultures, and Coagulation Biomarkers.
- **Prescriptions (Rx):** Broad-spectrum antibiotics (Meropenem, Ceftriaxone), vasopressors (Norepinephrine), anticoagulants (Heparin), antiplatelets (Aspirin, Ticagrelor), and bronchodilators (Albuterol).
- **Doctor Notes:** Cardiology notes, Pulmonology consults, Surgical recovery notes, Trauma ICU evaluations, and Neurology notes.
- **Discharge Summaries:** Post-operative hospital discharge summaries with outpatient follow-up.

### 2. Goal-Based Classification & Folder Routing
The agent evaluates incoming text using rule-based keywords:
- **Patient Extraction:** Matches patient IDs (`PAT-101` to `PAT-108`) or bed identifiers.
- **Department Routing:** Categorizes into *Cardiology*, *Pulmonology*, *Nephrology*, *Neurology*, *Infectious Disease*, *General Surgery*, or *Trauma ICU*.
- **Record Type Classification:** Categorizes into *Lab Report*, *Prescription*, *Doctor Note*, or *Discharge Summary*.

Files are organized into:
```text
data/organized_records/<Department>/<Record_Type>/<RecordID>_<PatientID>.txt
```

### 3. Allocation Audit Log Sample
```text
[ALLOCATED] REC-001 -> Patient: PAT-103 | Dept: Infectious Disease | Type: Lab Report   | Dest: organized_records/Infectious Disease/Lab Report
[ALLOCATED] REC-003 -> Patient: PAT-104 | Dept: Cardiology         | Type: Doctor Note  | Dest: organized_records/Cardiology/Doctor Note
[ALLOCATED] REC-005 -> Patient: PAT-107 | Dept: Pulmonology        | Type: Lab Report   | Dest: organized_records/Pulmonology/Lab Report
[ALLOCATED] REC-020 -> Patient: PAT-101 | Dept: General Surgery    | Type: Discharge Summary | Dest: organized_records/General Surgery/Discharge Summary
```

### 4. Multi-Attribute Search Engine
Search records by:
- Full-text clinical search (e.g. `"troponin"`, `"sepsis"`, `"ventilator"`).
- Filter by Patient ID (e.g. `PAT-103`).
- Filter by Department (e.g. `Cardiology`).
- Filter by Record Type (e.g. `Prescription`).

---

## 💻 Part 3: Interactive Streamlit Dashboard

Run with:
```bash
streamlit run app.py
```

### Key UI Features:
1. **Interactive Timeline Slider:** Slide from Hour 1 to Hour 24 to witness dynamic vital sign drift and see deteriorating patients trigger alerts.
2. **8 ICU Bed Telemetry Status Board:** Responsive grid of cards displaying live vital signs, diagnosis, risk category badge (Normal/Moderate/High), and risk score bars.
3. **Active Alerts Banner:** Prominently highlights active critical beds with current vitals and AI detection confidence.
4. **Patient Trajectory Inspector:** Select any bed to render the high-resolution 4-panel trajectory graph.
5. **Medical Record Browser & Search:** Real-time query search and dropdown filters, with expandable document cards to inspect clinical contents.
6. **Model Diagnostics Tab:** Interactive confusion matrix table and feature importance bar chart.

---

## ✅ Experimental Results & Verification

- **Code Validation:** Executed `python run_demo.py` with exit code `0`.
- **Streamlit Validation:** Dashboard rendered and verified via browser agent at `http://localhost:8501`.
- **False Alarm Rate:** Validated at **0.44%**, meeting clinical safety benchmarks by minimizing alert fatigue while catching 93.3% of critical deteriorations early.
