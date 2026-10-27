"""
Sample Medical Records Generator
================================
Creates 20 simulated medical records with realistic clinical content (no real patient data).
Record Types:
  - Lab Report
  - Prescription
  - Doctor Note
  - Discharge Summary
"""

import os
import json

SAMPLE_RECORDS = [
    {
        "record_id": "REC-001",
        "raw_text": """METROPOLITAN HOSPITAL - CLINICAL LABORATORY REPORT
Record ID: REC-001 | Date: 2026-10-25 08:30:00
Patient Identifier: PAT-103 (Bed 3)
Specimen: Blood Culture & CBC with Differential
Department: Infectious Disease / General ICU

TEST RESULTS:
- White Blood Cell (WBC): 18.5 x10^3/uL (HIGH, Ref: 4.5 - 11.0)
- Neutrophils: 88% (HIGH, Ref: 40 - 70)
- Serum Lactate: 3.8 mmol/L (HIGH, Ref: 0.5 - 2.0)
- Procalcitonin: 4.2 ng/mL (SEVERE SEPSIS RISK, Ref: < 0.5)
- Blood Culture: Gram-negative bacilli detected (Escherichia coli)

Lab Impression: High bacteremic burden consistent with systemic sepsis. Urgent antibiotic adjustment advised.""",
    },
    {
        "record_id": "REC-002",
        "raw_text": """METROPOLITAN HOSPITAL - ICU PHARMACY PRESCRIPTION
Record ID: REC-002 | Date: 2026-10-25 09:15:00
Patient: PAT-103 (Elena Vasquez) | Bed: 3
Attending Physician: Dr. K. Mehta, MD (Infectious Disease)

PRESCRIPTION ORDER (Rx):
1. Meropenem IV 1g every 8 hours via IV infusion over 30 mins
2. Norepinephrine bitartrate 4mg/250mL D5W, titrate MAP >= 65 mmHg
3. Normal Saline 0.9% 1000 mL bolus IV stat

Pharmacist Note: Monitor renal function and arterial pressure continuously.""",
    },
    {
        "record_id": "REC-003",
        "raw_text": """METROPOLITAN HOSPITAL - CARDIOLOGY CLINICAL PROGRESS NOTE
Record ID: REC-003 | Date: 2026-10-25 10:00:00
Patient: PAT-104 (David Kim) | Bed: 4 | Dept: Cardiology
Cardiologist: Dr. Laura Vance, MD, FACC

SUBJECTIVE / OBJECTIVE:
Patient status post acute non-ST elevation myocardial infarction (NSTEMI).
Complaining of retrosternal heaviness and diaphoresis.
ECG: Demonstrates dynamic ST-segment depressions in V4-V6 and inverted T waves.
Troponin-I elevated at 4.6 ng/mL (Normal < 0.04 ng/mL).
BP fluctuating around 90/60 mmHg; Heart Rate irregular at 112 bpm.

ASSESSMENT & PLAN:
1. Acute Coronary Syndrome with ongoing myocardial ischemia.
2. Transfer for emergent cardiac catheterization and coronary angiography.
3. Continue dual antiplatelet therapy and heparin infusion.""",
    },
    {
        "record_id": "REC-004",
        "raw_text": """METROPOLITAN HOSPITAL - CARDIAC DRUG PRESCRIPTION
Record ID: REC-004 | Date: 2026-10-25 10:45:00
Patient ID: PAT-104 | Bed 4 | Cardiology Division
Prescriber: Dr. Laura Vance, MD

MEDICATION ORDER (Rx):
1. Aspirin 81 mg chewable tablet once daily
2. Ticagrelor 90 mg PO BID
3. Heparin sodium IV bolus 5000 units followed by 1000 units/hr continuous infusion
4. Atorvastatin 80 mg oral at bedtime

Allergies: No known drug allergies (NKDA).""",
    },
    {
        "record_id": "REC-005",
        "raw_text": """METROPOLITAN HOSPITAL - ARTERIAL BLOOD GAS (ABG) LAB REPORT
Record ID: REC-005 | Date: 2026-10-25 11:20:00
Patient Ref: PAT-107 (Bed 7 - Maria Gonzalez)
Department: Pulmonology / Respiratory ICU

ARTERIAL BLOOD GAS PANEL (FiO2 60% via High-Flow Nasal Cannula):
- Arterial pH: 7.28 (LOW, Acidosis, Ref: 7.35 - 7.45)
- PaO2: 62 mmHg (SEVERE HYPOXEMIA, Ref: 80 - 100)
- PaCO2: 52 mmHg (ELEVATED, Ref: 35 - 45)
- HCO3-: 24 mEq/L (Normal: 22 - 26)
- SaO2: 86% (CRITICAL, Ref: > 94%)
- PaO2/FiO2 ratio: 103 (Moderate-to-Severe ARDS criteria)

Impression: Acute respiratory failure secondary to ARDS. Endotracheal intubation standby advised.""",
    },
    {
        "record_id": "REC-006",
        "raw_text": """METROPOLITAN HOSPITAL - PULMONOLOGY CONSULTATION NOTE
Record ID: REC-006 | Date: 2026-10-25 12:10:00
Patient: PAT-107 | Bed: 7 | Department: Pulmonology
Consulting Pulmonologist: Dr. Marcus Reynolds, MD

CLINICAL IMPRESSION:
71yo female with rapidly deteriorating pulmonary gas exchange.
Bilateral diffuse infiltrates on portable chest radiograph consistent with ARDS.
Severe tachypnea with respiratory rate > 30 bpm and intercostal retractions.

RECOMMENDATIONS:
- Immediate mechanical ventilation with lung-protective tidal volumes (6 mL/kg).
- Set PEEP to 12 cm H2O; adjust FiO2 to maintain SpO2 >= 90%.
- Prone positioning protocol if P/F ratio remains < 150.""",
    },
    {
        "record_id": "REC-007",
        "raw_text": """METROPOLITAN HOSPITAL - INPATIENT PRESCRIPTION
Record ID: REC-007 | Date: 2026-10-25 12:45:00
Patient: PAT-107 (Bed 7) | Pulmonology Unit
Ordering Clinician: Dr. Marcus Reynolds, MD

Rx ORDERS:
1. Propofol 10 mg/mL injectable emulsion, infuse at 20-50 mcg/kg/min for sedation
2. Fentanyl citrate 50 mcg/hr IV infusion for pain/ventilator synchrony
3. Methylprednisolone 40 mg IV every 12 hours
4. Albuterol sulfate 2.5 mg nebulized solution q4h PRN wheezing""",
    },
    {
        "record_id": "REC-008",
        "raw_text": """METROPOLITAN HOSPITAL - NEPHROLOGY LAB REPORT
Record ID: REC-008 | Date: 2026-10-25 13:30:00
Patient Identifier: PAT-106 (Bed 6)
Department: Nephrology / General ICU

COMPREHENSIVE RENAL & METABOLIC PANEL:
- Serum Creatinine: 2.4 mg/dL (ELEVATED, Baseline 0.9, Ref: 0.6 - 1.2)
- Blood Urea Nitrogen (BUN): 48 mg/dL (HIGH, Ref: 7 - 20)
- Estimated GFR: 28 mL/min/1.73m^2 (Stage 4 CKD / Acute Kidney Injury)
- Potassium: 5.6 mEq/L (MILD HYPERKALEMIA, Ref: 3.5 - 5.1)
- Sodium: 136 mEq/L (Normal)
- Serum Ketones: Positive 2+ (Resolving diabetic ketoacidosis)

Status: Acute kidney injury on resolving DKA background. Strict fluid balance required.""",
    },
    {
        "record_id": "REC-009",
        "raw_text": """METROPOLITAN HOSPITAL - NEUROLOGY CLINICAL NOTE
Record ID: REC-009 | Date: 2026-10-25 14:15:00
Patient: PAT-105 (Bed 5 - Amina Yusuf)
Department: Neurology / Neurocritical Care
Neurologist: Dr. S. Patel, MD

NEUROLOGICAL ASSESSMENT:
Patient admitted 18 hours ago post IV thrombolysis for left MCA ischemic stroke.
Glasgow Coma Scale: 14 (E4 V4 M6 - mild expressive aphasia).
NIHSS Stroke Scale score has improved from 12 on admission to 6 today.
Pupils equal and reactive to light, 3mm bilaterally.
No signs of hemorrhagic transformation on follow-up non-contrast head CT.

PLAN:
Maintain blood pressure < 140/90 mmHg.
Initiate oral statin and antiplatelet therapy tomorrow.
Speech and swallowing therapy consult ordered.""",
    },
    {
        "record_id": "REC-010",
        "raw_text": """METROPOLITAN HOSPITAL - NEUROLOGY MEDICATION ORDER
Record ID: REC-010 | Date: 2026-10-25 14:50:00
Patient ID: PAT-105 | Bed 5 | Dept: Neurology
Physician: Dr. S. Patel, MD

PRESCRIPTION (Rx):
1. Nicardipine IV infusion 5 mg/hr, titrate by 2.5 mg/hr q15m to target SBP 120-140 mmHg
2. Levetiracetam (Keppra) 500 mg IV BID for seizure prophylaxis
3. Subcutaneous Heparin 5000 units q12h for DVT prevention (start at hour 24 post-TPA)""",
    },
    {
        "record_id": "REC-011",
        "raw_text": """METROPOLITAN HOSPITAL - POST-OPERATIVE SURGICAL NOTE
Record ID: REC-011 | Date: 2026-10-25 15:30:00
Patient: PAT-101 (Bed 1 - Sarah Connor)
Department: General Surgery / ICU
Surgeon: Dr. Anthony Ross, FACS

SURGICAL RECOVERY NOTE:
Patient is Day 1 post laparoscopic appendectomy for acute perforated appendicitis.
Vitals remain stable: HR 72, BP 118/74, SpO2 98% on room air, Temp 36.8°C.
Surgical port sites clean, dry, and intact with no erythema or discharge.
Abdomen soft, mild peri-umbilical tenderness expected post-op. Bowel sounds present.

PLAN:
Advance from clear liquid diet to soft diet as tolerated.
Discontinue Foley catheter. Ambulate three times daily.
Anticipate discharge to surgical stepdown ward in 24 hours.""",
    },
    {
        "record_id": "REC-012",
        "raw_text": """METROPOLITAN HOSPITAL - POST-OP ANALGESIA PRESCRIPTION
Record ID: REC-012 | Date: 2026-10-25 16:00:00
Patient: PAT-101 (Bed 1) | Department: General Surgery
Prescriber: Dr. Anthony Ross, FACS

MEDICATION PRESCRIPTION (Rx):
1. Acetaminophen 650 mg IV q6h scheduled for 24 hours
2. Ondansetron 4 mg IV q8h PRN post-op nausea
3. Tramadol 50 mg oral q6h PRN breakthrough surgical pain
4. Cefazolin 1g IV q8h x 2 additional doses for surgical prophylaxis""",
    },
    {
        "record_id": "REC-013",
        "raw_text": """METROPOLITAN HOSPITAL - PULMONARY FUNCTION & SPUTUM LAB REPORT
Record ID: REC-013 | Date: 2026-10-25 16:45:00
Patient Ref: PAT-102 (Bed 2 - Robert Miller)
Department: Pulmonology
Lab Supervisor: H. Davis, MT(ASCP)

SPUTUM CULTURE & CYTOLOGY:
- Specimen: Purulent sputum
- Gram Stain: Many polymorphonuclear leukocytes, moderate Gram-positive diplococci
- Culture: Streptococcus pneumoniae sensitive to Ceftriaxone & Levofloxacin
- Nasal Swab: SARS-CoV-2 and Influenza A/B negative by PCR

Impression: Acute bacterial exacerbation of chronic bronchitis / COPD.""",
    },
    {
        "record_id": "REC-014",
        "raw_text": """METROPOLITAN HOSPITAL - PULMONOLOGY DOCTOR NOTE
Record ID: REC-014 | Date: 2026-10-25 17:15:00
Patient: PAT-102 | Bed 2 | Dept: Pulmonology
Attending: Dr. Elena Rostova, MD

PROGRESS NOTE:
68yo male with history of severe COPD, GOLD Stage III.
Currently alert, oriented x 3, in mild work of breathing.
Auscultation reveals scattered expiratory wheezes bilaterally with coarse rhonchi at right base.
SpO2 maintained at 92-94% on 2 L/min nasal cannula (safe target range for chronic CO2 retainer).

PLAN:
1. Continue intravenous Ceftriaxone and Azithromycin.
2. Inhaled bronchodilator therapy every 4 hours.
3. Chest physiotherapy and incentive spirometry.""",
    },
    {
        "record_id": "REC-015",
        "raw_text": """METROPOLITAN HOSPITAL - TRAUMA ICU PROGRESS NOTE
Record ID: REC-015 | Date: 2026-10-25 18:00:00
Patient: PAT-108 (Bed 8 - Thomas Wright)
Department: Trauma / General ICU
Attending Traumatologist: Dr. Gregory Hall, MD

CLINICAL EVALUATION:
29yo male admitted after motor vehicle collision with blunt thoracic trauma.
Follow-up eFAST ultrasound negative for pericardial tamponade or hemoperitoneum.
Serial hemoglobin stable at 13.8 g/dL.
Chest tube drainage < 50 mL serosanguinous fluid over last 8 hours; no air leak.

PLAN:
Clamp chest tube tomorrow morning if chest X-ray confirms full lung re-expansion.
Multimodal pain management with oral analgesics.
Physical therapy evaluation for mobilization.""",
    },
    {
        "record_id": "REC-016",
        "raw_text": """METROPOLITAN HOSPITAL - CARDIOLOGY ECHOCARDIOGRAM REPORT
Record ID: REC-016 | Date: 2026-10-25 18:40:00
Patient: PAT-104 (Bed 4) | Department: Cardiology
Director of Echocardiography: Dr. Laura Vance, MD

TRANSTHORACIC ECHOCARDIOGRAPHY (TTE) LAB FINDINGS:
- Left Ventricular Ejection Fraction (LVEF): 38% (MODERATELY REDUCED)
- Regional Wall Motion Abnormality: Severe hypokinesis of anterior and apical walls
- Diastolic Filling: Grade II diastolic dysfunction
- Right Ventricular Function: Normal systolic excursion (TAPSE 19 mm)
- Valves: Trace mitral regurgitation, no aortic stenosis

Summary: Antero-apical myocardial injury with depressed left ventricular systolic function.""",
    },
    {
        "record_id": "REC-017",
        "raw_text": """METROPOLITAN HOSPITAL - NEPHROLOGY CONSULTATION NOTE
Record ID: REC-017 | Date: 2026-10-25 19:15:00
Patient: PAT-106 (James Henderson - Bed 6)
Department: Nephrology
Nephrologist: Dr. Arthur Chen, MD

CONSULTATION SUMMARY:
Patient with acute tubular necrosis (ATN) superimposed on diabetic nephropathy.
Urine output over past 6 hours has improved to 45 mL/hr following fluid resuscitation.
Electrolytes normalizing; metabolic acidosis resolving with intravenous bicarbonate infusion.

RECOMMENDATIONS:
- Discontinue nephrotoxic agents (NSAIDs, aminoglycosides).
- Maintain intravenous hydration with 0.45% Saline at 100 mL/hr.
- Re-check serum creatinine and potassium in 12 hours.""",
    },
    {
        "record_id": "REC-018",
        "raw_text": """METROPOLITAN HOSPITAL - INFECTIOUS DISEASE LAB REPORT
Record ID: REC-018 | Date: 2026-10-25 19:50:00
Patient ID: PAT-103 (Bed 3)
Department: Infectious Disease / General ICU

SERIAL COAGULATION & INFLAMMATORY BIOMARKERS:
- D-Dimer: 4,800 ng/mL FEU (CRITICALLY HIGH, Ref: < 500)
- Fibrinogen: 160 mg/dL (LOW / CONSUMPTIVE COAGULOPATHY, Ref: 200 - 400)
- International Normalized Ratio (INR): 1.85 (ELEVATED, Ref: 0.9 - 1.1)
- Platelet Count: 68 x10^3/uL (THROMBOCYTOPENIA, Ref: 150 - 450)

Clinical Impression: Emerging Disseminated Intravascular Coagulation (DIC) in setting of severe urosepsis.""",
    },
    {
        "record_id": "REC-019",
        "raw_text": """METROPOLITAN HOSPITAL - PULMONOLOGY RESPIRATORY PRESCRIPTION
Record ID: REC-019 | Date: 2026-10-25 20:25:00
Patient: PAT-102 (Robert Miller - Bed 2)
Department: Pulmonology
Prescriber: Dr. Elena Rostova, MD

PRESCRIPTION (Rx):
1. Ipratropium bromide / Albuterol (Combivent) inhalation aerosol 2 puffs QID
2. Budesonide 0.5 mg nebulized suspension BID
3. Prednisone 40 mg oral tablet once daily for 5-day taper
4. Supplemental nasal oxygen at 2 L/min continuous via nasal cannula""",
    },
    {
        "record_id": "REC-020",
        "raw_text": """METROPOLITAN HOSPITAL - FORMAL DISCHARGE SUMMARY
Record ID: REC-020 | Date: 2026-10-25 21:00:00
Patient: PAT-101 (Sarah Connor - Bed 1)
Department: General Surgery
Attending Surgeon: Dr. Anthony Ross, FACS

DISCHARGE SUMMARY:
Admission Date: 2026-10-23 | Discharge Date: 2026-10-26
Primary Diagnosis: Acute Perforated Appendicitis
Procedure: Laparoscopic Appendectomy with peritoneal irrigation

Hospital Course:
The patient experienced an uneventful recovery with stable vital signs throughout the post-operative period.
Afebrile with excellent pain control on oral analgesics. Tolerating regular oral intake.
Surgical incisions healed without complication.

Discharge Instructions & Follow-up:
- Follow up in Surgical Outpatient Clinic in 10 days for wound check.
- Avoid strenuous physical exertion or lifting > 10 lbs for 2 weeks.
- Return to ED immediately if fever > 38.5°C, worsening abdominal pain, or nausea develops.""",
    }
]


def create_sample_record_files(dest_dir: str = "data/unallocated_records") -> list:
    """
    Writes all 20 sample medical records to raw text files in the destination directory.
    Returns list of file paths.
    """
    os.makedirs(dest_dir, exist_ok=True)
    created_paths = []

    for record in SAMPLE_RECORDS:
        rec_id = record["record_id"]
        file_path = os.path.join(dest_dir, f"{rec_id}.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(record["raw_text"])
        created_paths.append(file_path)

    return created_paths


if __name__ == "__main__":
    paths = create_sample_record_files()
    print(f"Created {len(paths)} unallocated medical records in data/unallocated_records/")
