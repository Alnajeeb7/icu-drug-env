"""
Synthetic ICU patient generator.
Modeled from MIMIC-III demographic distributions (public knowledge).
No external data dependency — fully self-contained.
"""

import random
import uuid
from typing import Dict, Any

from env.models import PatientProfile, PatientVitals


DIAGNOSES = [
    "septic shock",
    "pneumonia",
    "acute respiratory failure",
    "acute kidney injury",
    "diabetic ketoacidosis",
    "acute myocardial infarction",
    "congestive heart failure",
    "liver failure",
    "post-operative care",
    "drug overdose",
]

COMORBIDITIES_POOL = [
    "hypertension",
    "type 2 diabetes",
    "chronic kidney disease",
    "atrial fibrillation",
    "COPD",
    "coronary artery disease",
    "heart failure",
    "obesity",
    "hypothyroidism",
    "liver cirrhosis",
]

ALLERGIES_POOL = [
    "penicillin",
    "sulfa drugs",
    "contrast dye",
    "codeine",
    "latex",
    "NKDA",
    "NKDA",
    "NKDA",
]

RENAL_STATES = [
    "normal",
    "mild_impairment",
    "moderate_impairment",
    "severe_impairment",
    "dialysis",
]

RENAL_CREATININE = {
    "normal": (0.6, 1.2),
    "mild_impairment": (1.3, 1.9),
    "moderate_impairment": (2.0, 3.4),
    "severe_impairment": (3.5, 7.9),
    "dialysis": (4.0, 12.0),
}


def generate_patient(
    seed: int = None,
    renal_state: str = None,
    diagnosis: str = None,
) -> PatientProfile:
    rng = random.Random(seed)
    age = rng.randint(35, 85)
    weight_kg = round(rng.uniform(50, 120), 1)
    sex = rng.choice(["male", "female"])
    selected_diagnosis = diagnosis or rng.choice(DIAGNOSES)
    num_comorbidities = rng.randint(1, 4)
    comorbidities = rng.sample(COMORBIDITIES_POOL, k=num_comorbidities)
    allergy = rng.choice(ALLERGIES_POOL)
    renal = renal_state or rng.choice(RENAL_STATES)
    hepatic = rng.choices(
        ["normal", "mild", "moderate", "severe"],
        weights=[70, 15, 10, 5],
    )[0]

    meds_pool = [
        "lisinopril", "metoprolol", "aspirin", "atorvastatin",
        "pantoprazole", "insulin_regular", "furosemide",
    ]
    num_meds = rng.randint(0, 3)
    current_meds = rng.sample(meds_pool, k=num_meds)

    return PatientProfile(
        patient_id=str(uuid.uuid4())[:8],
        age=age,
        weight_kg=weight_kg,
        sex=sex,
        diagnosis=selected_diagnosis,
        comorbidities=comorbidities,
        allergies=[allergy],
        renal_function=renal,
        hepatic_function=hepatic,
        current_medications=current_meds,
    )


def generate_vitals(
    patient: PatientProfile,
    seed: int = None,
    severity: str = "moderate",
) -> PatientVitals:
    rng = random.Random(seed)

    creatinine_range = RENAL_CREATININE[patient.renal_function]
    creatinine = round(rng.uniform(*creatinine_range), 2)

    if severity == "critical":
        hr = rng.randint(110, 135)
        sbp = rng.randint(70, 90)
        dbp = rng.randint(40, 60)
        rr = rng.randint(22, 30)
        spo2 = rng.randint(88, 93)
        glucose = rng.randint(200, 380)
        temp = round(rng.uniform(38.5, 40.0), 1)
    elif severity == "moderate":
        hr = rng.randint(90, 115)
        sbp = rng.randint(85, 110)
        dbp = rng.randint(50, 70)
        rr = rng.randint(18, 24)
        spo2 = rng.randint(93, 97)
        glucose = rng.randint(150, 250)
        temp = round(rng.uniform(37.5, 39.0), 1)
    else:
        hr = rng.randint(70, 95)
        sbp = rng.randint(100, 130)
        dbp = rng.randint(60, 85)
        rr = rng.randint(14, 20)
        spo2 = rng.randint(95, 99)
        glucose = rng.randint(100, 160)
        temp = round(rng.uniform(36.5, 37.5), 1)

    map_val = round((sbp + 2 * dbp) / 3, 1)
    potassium = round(rng.uniform(3.2, 5.2), 1)
    sodium = round(rng.uniform(133, 147), 1)
    inr = None
    if "warfarin" in patient.current_medications:
        inr = round(rng.uniform(1.5, 4.5), 1)

    return PatientVitals(
        heart_rate=float(hr),
        systolic_bp=float(sbp),
        diastolic_bp=float(dbp),
        map=map_val,
        respiratory_rate=float(rr),
        spo2=float(spo2),
        temperature=temp,
        glucose=float(glucose),
        creatinine=creatinine,
        potassium=potassium,
        sodium=sodium,
        inr=inr,
    )


def evolve_vitals(
    vitals: PatientVitals,
    prescription: Dict[str, Any],
    seed: int = None,
) -> PatientVitals:
    """
    Simulate how vitals change based on current prescription.
    Used in the hard task to simulate time progression.
    """
    rng = random.Random(seed)
    drugs = list(prescription.keys()) if prescription else []

    hr = vitals.heart_rate
    sbp = vitals.systolic_bp
    dbp = vitals.diastolic_bp
    glucose = vitals.glucose
    creatinine = vitals.creatinine
    spo2 = vitals.spo2
    potassium = vitals.potassium

    # --- Vasopressors ---
    if "norepinephrine" in drugs:
        sbp = min(sbp + rng.uniform(5, 15), 160)
        dbp = min(dbp + rng.uniform(3, 8), 100)
        hr = max(hr - rng.uniform(0, 5), 60)
    elif "vasopressin" in drugs:
        sbp = min(sbp + rng.uniform(3, 10), 150)
        dbp = min(dbp + rng.uniform(2, 6), 95)
    elif sbp < 80:
        sbp = max(sbp - rng.uniform(2, 8), 50)

    # --- Glucose management ---
    if "insulin_regular" in drugs and glucose > 150:
        drop = rng.uniform(20, 60)
        glucose = max(glucose - drop, 60)
    elif glucose > 200:
        glucose = min(glucose + rng.uniform(5, 20), 500)

    # --- Steroids increase glucose ---
    if "dexamethasone" in drugs or "hydrocortisone" in drugs:
        glucose = min(glucose + rng.uniform(10, 35), 450)

    # --- Diuretics ---
    if "furosemide" in drugs:
        potassium = max(potassium - rng.uniform(0.1, 0.5), 2.0)

    # --- Potassium replacement ---
    if "potassium_chloride" in drugs:
        potassium = min(potassium + rng.uniform(0.2, 0.6), 6.5)

    # --- Nephrotoxicity ---
    if "piperacillin_tazobactam" in drugs and "vancomycin" in drugs:
        creatinine = min(creatinine + rng.uniform(0.1, 0.4), 8.0)

    # --- Sedation effects ---
    if "propofol" in drugs:
        sbp = max(sbp - rng.uniform(3, 10), 60)
        hr = max(hr - rng.uniform(2, 8), 45)
    if "fentanyl" in drugs or "midazolam" in drugs:
        spo2 = max(spo2 - rng.uniform(0, 2), 85)
        hr = max(hr - rng.uniform(0, 4), 45)

    # --- Random drift ---
    hr += rng.uniform(-5, 5)
    hr = max(40, min(160, round(hr, 0)))
    sbp = max(60, min(200, round(sbp, 0)))
    dbp = max(40, min(120, round(dbp, 0)))
    map_val = round((sbp + 2 * dbp) / 3, 1)
    glucose = max(40, min(500, round(glucose, 0)))
    creatinine = max(0.3, min(12.0, round(creatinine, 2)))
    spo2 = max(80, min(100, round(spo2 + rng.uniform(-1, 1), 1)))
    potassium = max(2.0, min(7.0, round(potassium + rng.uniform(-0.1, 0.1), 1)))

    return PatientVitals(
        heart_rate=float(hr),
        systolic_bp=float(sbp),
        diastolic_bp=float(dbp),
        map=map_val,
        respiratory_rate=float(max(8, min(35, vitals.respiratory_rate + rng.uniform(-1, 1)))),
        spo2=float(spo2),
        temperature=float(round(max(35.0, min(41.0, vitals.temperature + rng.uniform(-0.1, 0.2))), 1)),
        glucose=float(glucose),
        creatinine=float(creatinine),
        potassium=float(potassium),
        sodium=float(round(max(120, min(160, vitals.sodium + rng.uniform(-1, 1))), 1)),
        inr=vitals.inr,
    )
