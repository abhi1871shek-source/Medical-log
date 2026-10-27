"""
Medical Log Allocation & Rule-Based Classifier (Goal-Based Agent)
=================================================================
AI Agent Type: Goal-Based Agent
  The agent operates with a specific goal:
  "Classify all unallocated medical records by Patient ID, Department,
   and Record Type using rule-based keyword matching, store them in organized
   directory hierarchies, update an index catalog, and maintain an audit log."

Features:
  - Rule-based keyword matching for Patient ID, Department, and Record Type
  - Automatic directory creation and file storage
  - Audit logging for every record processed
  - Full-text & multi-criteria search function (by Patient, Dept, Type, or Keywords)
"""

import os
import re
import shutil
import json
from typing import Dict, List, Optional


# -------------------------------------------------------------
# Keyword Taxonomy & Classification Rules
# -------------------------------------------------------------
PATIENT_BED_MAP = {
    "1": "PAT-101",
    "2": "PAT-102",
    "3": "PAT-103",
    "4": "PAT-104",
    "5": "PAT-105",
    "6": "PAT-106",
    "7": "PAT-107",
    "8": "PAT-108"
}

DEPARTMENT_RULES = {
    "Cardiology": [
        "cardiology", "troponin", "ecg", "myocardial", "echocardiogram",
        "infarction", "angina", "coronary", "catheterization", "ticagrelor", "stemi", "nstem"
    ],
    "Pulmonology": [
        "pulmonology", "abg", "arterial blood gas", "pao2", "copd", "sputum",
        "respiratory", "bronchitis", "ards", "ventilator", "albuterol", "peep", "hypoxemia"
    ],
    "Nephrology": [
        "nephrology", "creatinine", "bun", "dialysis", "gfr", "renal",
        "kidney", "tubular necrosis", "hyperkalemia", "nephropathy"
    ],
    "Neurology": [
        "neurology", "neurocritical", "stroke", "nihss", "glasgow", "gcs",
        "nicardipine", "tpa", "thrombolysis", "head ct", "aphasia", "keppra", "brain"
    ],
    "Infectious Disease": [
        "urosepsis", "sepsis", "procalcitonin", "blood culture", "meropenem",
        "lactate", "bacteremic", "dic", "disseminated intravascular", "infection"
    ],
    "General Surgery": [
        "appendicitis", "appendectomy", "surgical", "post-op", "laparoscopic",
        "abdomen", "wound", "bowel", "perforated"
    ],
    "Trauma ICU": [
        "trauma", "efast", "collision", "chest tube", "tamponade",
        "hemoperitoneum", "blunt thoracic"
    ]
}

RECORD_TYPE_RULES = {
    "Discharge Summary": [
        "discharge summary", "discharge instructions", "hospital course",
        "admission date:", "discharge date:"
    ],
    "Prescription": [
        "prescription", "rx orders", "prescription order", "medication prescription",
        "medication order", "pharmacy prescription", "rx:"
    ],
    "Lab Report": [
        "laboratory report", "lab report", "test results:", "arterial blood gas panel",
        "specimen:", "renal & metabolic panel", "lab findings", "lab impression"
    ],
    "Doctor Note": [
        "clinical progress note", "consultation note", "clinical evaluation",
        "surgical recovery note", "progress note", "doctor note", "assessment & plan",
        "clinical impression", "neurological assessment"
    ]
}


class MedicalLogAllocatorAgent:
    """
    Goal-Based Agent for Medical Log Allocation & Classification.
    """

    def __init__(
        self,
        unallocated_dir: str = "data/unallocated_records",
        organized_dir: str = "data/organized_records"
    ):
        self.unallocated_dir = unallocated_dir
        self.organized_dir = organized_dir
        self.catalog: List[Dict] = []
        self.allocation_logs: List[str] = []

    def classify_record_text(self, text: str, fallback_id: str = "REC-UNKNOWN") -> Dict:
        """
        Applies rule-based keyword matching to extract Patient ID, Department, and Record Type.
        """
        text_lower = text.lower()

        # 1. Detect Patient ID
        patient_id = "UNKNOWN_PATIENT"
        pat_match = re.search(r"\b(PAT-\d{3})\b", text, re.IGNORECASE)
        if pat_match:
            patient_id = pat_match.group(1).upper()
        else:
            # Fallback to Bed number lookup
            bed_match = re.search(r"Bed\s*[:\s#]?\s*(\d)", text, re.IGNORECASE)
            if bed_match and bed_match.group(1) in PATIENT_BED_MAP:
                patient_id = PATIENT_BED_MAP[bed_match.group(1)]

        # 2. Detect Department
        dept_scores = {}
        for dept, keywords in DEPARTMENT_RULES.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                dept_scores[dept] = score

        if dept_scores:
            assigned_department = max(dept_scores, key=dept_scores.get)
        else:
            assigned_department = "General ICU"

        # 3. Detect Record Type (Prioritized: Discharge Summary -> Prescription -> Lab Report -> Doctor Note)
        assigned_type = "Doctor Note"  # default clinical document
        type_scores = {}
        for rtype, keywords in RECORD_TYPE_RULES.items():
            score = sum(2 if kw in text_lower else 0 for kw in keywords)
            if score > 0:
                type_scores[rtype] = score

        if type_scores:
            assigned_type = max(type_scores, key=type_scores.get)

        # 4. Extract Record ID
        rec_match = re.search(r"\b(REC-\d{3})\b", text, re.IGNORECASE)
        record_id = rec_match.group(1).upper() if rec_match else fallback_id

        return {
            "record_id": record_id,
            "patient_id": patient_id,
            "department": assigned_department,
            "record_type": assigned_type,
            "confidence": "High (Rule-Matched)"
        }

    def allocate_all_records(self) -> List[Dict]:
        """
        Goal-directed action:
        Scans all files in `unallocated_dir`, classifies them, creates target
        department/type folder hierarchies, saves structured copies, and logs results.
        """
        if not os.path.exists(self.unallocated_dir):
            raise FileNotFoundError(f"Unallocated directory '{self.unallocated_dir}' does not exist.")

        self.catalog.clear()
        self.allocation_logs.clear()

        record_files = sorted([f for f in os.listdir(self.unallocated_dir) if f.endswith(".txt")])

        for fname in record_files:
            file_path = os.path.join(self.unallocated_dir, fname)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            base_name = os.path.splitext(fname)[0]
            classification = self.classify_record_text(content, fallback_id=base_name)

            rec_id = classification["record_id"]
            pat_id = classification["patient_id"]
            dept = classification["department"]
            rtype = classification["record_type"]

            # Define clean target storage path: organized_records/<Department>/<Record_Type>/
            dest_folder = os.path.join(self.organized_dir, dept, rtype)
            os.makedirs(dest_folder, exist_ok=True)

            dest_filename = f"{rec_id}_{pat_id}.txt"
            dest_path = os.path.join(dest_folder, dest_filename)

            # Write organized file with metadata header
            with open(dest_path, "w", encoding="utf-8") as f:
                f.write(f"--- ALLOCATION METADATA ---\n")
                f.write(f"RECORD_ID:   {rec_id}\n")
                f.write(f"PATIENT_ID:  {pat_id}\n")
                f.write(f"DEPARTMENT:  {dept}\n")
                f.write(f"RECORD_TYPE: {rtype}\n")
                f.write(f"STORED_PATH: {dest_path}\n")
                f.write(f"---------------------------\n\n")
                f.write(content)

            # Record in catalog
            entry = {
                "record_id": rec_id,
                "patient_id": pat_id,
                "department": dept,
                "record_type": rtype,
                "file_path": dest_path,
                "relative_path": os.path.relpath(dest_path, start="."),
                "content_preview": content[:160].replace("\n", " "),
                "full_text": content
            }
            self.catalog.append(entry)

            # Print & store allocation log
            log_msg = f"[ALLOCATED] {rec_id} -> Patient: {pat_id:7s} | Dept: {dept:18s} | Type: {rtype:17s} | Dest: {dest_folder}"
            self.allocation_logs.append(log_msg)
            print(log_msg)

        # Save catalog index as JSON for quick indexing
        index_path = os.path.join(self.organized_dir, "allocation_index.json")
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(
                [{k: v for k, v in e.items() if k != "full_text"} for e in self.catalog],
                f,
                indent=2
            )

        return self.catalog

    def search_records(
        self,
        query: Optional[str] = None,
        patient_id: Optional[str] = None,
        department: Optional[str] = None,
        record_type: Optional[str] = None
    ) -> List[Dict]:
        """
        Multi-attribute search across the organized records.
        Supports keyword full-text query, patient ID, department, and record type filters.
        """
        results = []
        q = query.lower().strip() if query else None
        p_filter = patient_id.upper().strip() if (patient_id and patient_id != "All") else None
        d_filter = department.strip() if (department and department != "All") else None
        t_filter = record_type.strip() if (record_type and record_type != "All") else None

        for item in self.catalog:
            # Filter checks
            if p_filter and item["patient_id"] != p_filter:
                continue
            if d_filter and item["department"] != d_filter:
                continue
            if t_filter and item["record_type"] != t_filter:
                continue
            if q:
                # Text match against record_id, department, record_type, or full content
                searchable_text = f"{item['record_id']} {item['patient_id']} {item['department']} {item['record_type']} {item['full_text']}".lower()
                if q not in searchable_text:
                    continue

            results.append(item)

        return results
