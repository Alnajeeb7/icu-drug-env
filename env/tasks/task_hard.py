"""
Task 3 (Hard): Full ICU Drug Management
Agent manages a critically ill patient over 10 steps.
Vitals change each step based on prescribed drugs.
Agent must keep all vitals in safe range.
Score = rolling average of vitals-in-range over all steps.
"""

import random
from typing import Dict, Any, Tuple, List

from env.drug_data import DANGEROUS_INTERACTIONS, DRUG_DB, VITALS_RANGES
from env.models import PatientProfile, PatientVitals, ICUObservation
from env.patient_generator import generate_patient, generate_vitals, evolve_vitals

TASK_NAME = "icu_management"
MAX_STEPS = 10

ICU_SCENARIOS = [
    {
        "diagnosis": "septic shock with multi-organ dysfunction",
        "initial_drugs": ["norepinephrine", "piperacillin_tazobactam", "vancomycin"],
        "available_to_prescribe": [
            "norepinephrine", "vasopressin", "dopamine", "epinephrine",
            "piperacillin_tazobactam", "vancomycin", "metronidazole",
            "insulin_regular", "furosemide", "potassium_chloride",
            "fentanyl", "propofol", "midazolam", "lorazepam",
            "pantoprazole", "ondansetron", "dexamethasone", "hydrocortisone",
            "heparin", "acetaminophen",
        ],
        "target_vitals": {
            "map": (65, 100),
            "glucose": (140, 180),
            "heart_rate": (60, 100),
            "spo2": (94, 100),
            "potassium": (3.5, 5.0),
        },
    },
    {
        "diagnosis": "acute respiratory distress syndrome (ARDS)",
        "initial_drugs": ["norepinephrine", "piperacillin_tazobactam", "fentanyl"],
        "available_to_prescribe": [
            "norepinephrine", "vasopressin", "epinephrine",
            "piperacillin_tazobactam", "vancomycin", "metronidazole",
            "insulin_regular", "furosemide", "potassium_chloride",
            "fentanyl", "propofol", "midazolam", "lorazepam",
            "pantoprazole", "dexamethasone", "hydrocortisone",
            "heparin", "acetaminophen", "ondansetron",
        ],
        "target_vitals": {
            "map": (65, 90),
            "glucose": (140, 180),
            "heart_rate": (60, 100),
            "spo2": (92, 100),
            "potassium": (3.5, 5.0),
        },
    },
    {
        "diagnosis": "diabetic ketoacidosis with sepsis",
        "initial_drugs": ["insulin_regular", "piperacillin_tazobactam", "potassium_chloride"],
        "available_to_prescribe": [
            "norepinephrine", "vasopressin",
            "piperacillin_tazobactam", "vancomycin", "metronidazole",
            "insulin_regular", "furosemide", "potassium_chloride",
            "fentanyl", "propofol", "midazolam",
            "pantoprazole", "ondansetron", "dexamethasone", "hydrocortisone",
            "heparin", "acetaminophen",
        ],
        "target_vitals": {
            "map": (65, 100),
            "glucose": (150, 200),
            "heart_rate": (60, 110),
            "spo2": (94, 100),
            "potassium": (3.5, 5.2),
        },
    },
    {
        "diagnosis": "post-cardiac arrest syndrome",
        "initial_drugs": ["norepinephrine", "fentanyl", "propofol"],
        "available_to_prescribe": [
            "norepinephrine", "vasopressin", "dopamine", "epinephrine",
            "piperacillin_tazobactam", "vancomycin",
            "insulin_regular", "furosemide", "potassium_chloride",
            "fentanyl", "propofol", "midazolam", "lorazepam",
            "pantoprazole", "amiodarone", "hydrocortisone",
            "heparin", "acetaminophen", "levetiracetam",
        ],
        "target_vitals": {
            "map": (70, 100),
            "glucose": (140, 180),
            "heart_rate": (50, 100),
            "spo2": (94, 100),
            "potassium": (3.5, 4.5),
        },
    },
    {
        "diagnosis": "acute liver failure with coagulopathy",
        "initial_drugs": ["pantoprazole", "vitamin_k", "acetaminophen"],
        "available_to_prescribe": [
            "norepinephrine", "vasopressin",
            "piperacillin_tazobactam", "vancomycin", "metronidazole",
            "insulin_regular", "furosemide", "potassium_chloride",
            "fentanyl", "midazolam",
            "pantoprazole", "ondansetron", "dexamethasone",
            "vitamin_k", "acetaminophen", "lorazepam",
        ],
        "target_vitals": {
            "map": (65, 95),
            "glucose": (100, 180),
            "heart_rate": (60, 100),
            "spo2": (94, 100),
            "potassium": (3.5, 5.0),
        },
    },
]


def get_scenario(seed: int = None) -> Dict[str, Any]:
    rng = random.Random(seed)
    return dict(rng.choice(ICU_SCENARIOS))


def build_observation(
    patient: PatientProfile,
    vitals: PatientVitals,
    scenario: Dict[str, Any],
    step: int,
    current_prescription: Dict[str, Any],
    feedback: str = None,
) -> ICUObservation:
    alerts = _generate_alerts(vitals, current_prescription)

    instruction = (
        f"TASK: Manage this critically ill ICU patient — Step {step}/{MAX_STEPS}.\n"
        f"Patient: {patient.weight_kg}kg, {patient.age}yo, {patient.sex}\n"
        f"Diagnosis: {scenario['diagnosis']}\n"
        f"Renal function: {patient.renal_function}\n\n"
        f"CURRENT VITALS (targets in parentheses):\n"
        f"  MAP: {vitals.map} mmHg (target: 65-100)\n"
        f"  Heart Rate: {vitals.heart_rate} bpm (target: 60-100)\n"
        f"  SpO2: {vitals.spo2}% (target: >94%)\n"
        f"  Glucose: {vitals.glucose} mg/dL (target: 140-180)\n"
        f"  Potassium: {vitals.potassium} mEq/L (target: 3.5-5.0)\n"
        f"  Creatinine: {vitals.creatinine} mg/dL\n\n"
        f"CURRENT PRESCRIPTION: {list(current_prescription.keys()) or 'None'}\n\n"
        f"Respond with action_type='adjust_dose' or 'prescribe_dose'.\n"
        f"Specify drug_name and dose_mg (or use 'hold_drug' to discontinue).\n"
        f"Provide rationale for your decision.\n"
        f"ALERTS: {'; '.join(alerts) if alerts else 'None'}"
    )

    lab_rng = random.Random(step * 7 + 13)
    return ICUObservation(
        task_name=TASK_NAME,
        step=step,
        patient=patient,
        vitals=vitals,
        available_drugs=scenario["available_to_prescribe"],
        current_prescription=current_prescription,
        alert_flags=alerts,
        lab_results={
            "creatinine": vitals.creatinine,
            "glucose": vitals.glucose,
            "potassium": vitals.potassium,
            "sodium": vitals.sodium,
            "wbc": round(lab_rng.uniform(8, 25), 1),
            "lactate": round(lab_rng.uniform(1.5, 6.0), 1),
        },
        task_instruction=instruction,
        previous_action_feedback=feedback,
    )


def _generate_alerts(
    vitals: PatientVitals,
    prescription: Dict[str, Any],
) -> List[str]:
    alerts = []
    if vitals.map < 65:
        alerts.append(f"CRITICAL: MAP {vitals.map} mmHg — below target 65 mmHg. Increase vasopressor.")
    if vitals.glucose > 250:
        alerts.append(f"CRITICAL: Glucose {vitals.glucose} mg/dL — initiate/increase insulin infusion.")
    if vitals.glucose < 70:
        alerts.append(f"CRITICAL: Hypoglycemia {vitals.glucose} mg/dL — hold insulin, give D50.")
    if vitals.potassium < 3.0:
        alerts.append(f"CRITICAL: Hypokalemia K+ {vitals.potassium} — replace potassium IV.")
    if vitals.potassium > 6.0:
        alerts.append(f"CRITICAL: Hyperkalemia K+ {vitals.potassium} — stop KCl, treat urgently.")
    if vitals.spo2 < 90:
        alerts.append(f"CRITICAL: SpO2 {vitals.spo2}% — increase oxygen support.")
    if vitals.heart_rate > 130:
        alerts.append(f"WARNING: Tachycardia HR {vitals.heart_rate} — assess for cause.")
    if vitals.creatinine > 3.0:
        alerts.append(f"WARNING: AKI — Creatinine {vitals.creatinine}. Adjust renally-cleared drugs.")

    drugs = list(prescription.keys())
    for i, d1 in enumerate(drugs):
        for d2 in drugs[i + 1:]:
            pair = (d1, d2)
            pair_rev = (d2, d1)
            if pair in DANGEROUS_INTERACTIONS:
                info = DANGEROUS_INTERACTIONS[pair]
                if info["severity"] in ("major", "contraindicated"):
                    alerts.append(f"DRUG INTERACTION: {d1} + {d2} — {info['effect']}")
            elif pair_rev in DANGEROUS_INTERACTIONS:
                info = DANGEROUS_INTERACTIONS[pair_rev]
                if info["severity"] in ("major", "contraindicated"):
                    alerts.append(f"DRUG INTERACTION: {d1} + {d2} — {info['effect']}")

    return alerts


def score_vitals(vitals: PatientVitals, scenario: Dict[str, Any], prescription: Dict[str, Any] = None) -> Tuple[float, Dict[str, float]]:
    """
    Score how well vitals are maintained.
    Returns overall score and per-vital breakdown.
    """
    targets = scenario["target_vitals"]
    scores = {}

    vital_values = {
        "map": vitals.map,
        "glucose": vitals.glucose,
        "heart_rate": vitals.heart_rate,
        "spo2": vitals.spo2,
        "potassium": vitals.potassium,
    }

    for vital_name, (low, high) in targets.items():
        value = vital_values.get(vital_name, 0)
        if low <= value <= high:
            scores[vital_name] = 1.0
        else:
            if value < low:
                deviation = (low - value) / low
            else:
                deviation = (value - high) / high
            scores[vital_name] = max(0.0, 1.0 - deviation)

    drug_interaction_penalty = 0.0
    drugs = list(prescription.keys()) if prescription else []
    for i, d1 in enumerate(drugs):
        for d2 in drugs[i + 1:]:
            pair = (d1, d2)
            pair_rev = (d2, d1)
            key = pair if pair in DANGEROUS_INTERACTIONS else (pair_rev if pair_rev in DANGEROUS_INTERACTIONS else None)
            if key and DANGEROUS_INTERACTIONS[key]["severity"] == "contraindicated":
                drug_interaction_penalty = 0.3
                break

    overall = max(0.0, sum(scores.values()) / len(scores) - drug_interaction_penalty)
    return round(overall, 4), scores


def grade_action(
    action: Dict[str, Any],
    vitals: PatientVitals,
    prescription: Dict[str, Any],
    scenario: Dict[str, Any],
    step: int,
    step_scores: List[float],
) -> Tuple[float, str, Dict[str, Any], bool]:
    """
    Returns (step_reward, feedback, updated_prescription, done).
    """
    action_type = action.get("action_type", "")
    drug = action.get("drug_name", "")
    dose = action.get("dose_mg")

    updated_prescription = dict(prescription)

    if action_type == "prescribe_dose" and drug:
        updated_prescription[drug] = {"dose_mg": dose, "status": "active"}
    elif action_type == "adjust_dose" and drug:
        if drug in updated_prescription:
            updated_prescription[drug]["dose_mg"] = dose
        else:
            updated_prescription[drug] = {"dose_mg": dose, "status": "active"}
    elif action_type == "hold_drug" and drug:
        updated_prescription.pop(drug, None)

    vital_score, vital_breakdown = score_vitals(vitals, scenario, updated_prescription)

    interaction_penalty = 0.0
    drugs_active = list(updated_prescription.keys())
    for i, d1 in enumerate(drugs_active):
        for d2 in drugs_active[i + 1:]:
            pair = (d1, d2)
            pair_rev = (d2, d1)
            key = pair if pair in DANGEROUS_INTERACTIONS else (pair_rev if pair_rev in DANGEROUS_INTERACTIONS else None)
            if key:
                info = DANGEROUS_INTERACTIONS[key]
                if info["severity"] == "contraindicated":
                    interaction_penalty += 0.4
                elif info["severity"] == "major":
                    interaction_penalty += 0.2

    step_reward = max(0.0, vital_score - interaction_penalty)
    step_reward = round(step_reward, 4)

    feedback_parts = [f"Step {step} vitals score: {vital_score:.2f}"]
    for v, s in vital_breakdown.items():
        feedback_parts.append(f"  {v}: {s:.2f}")
    if interaction_penalty > 0:
        feedback_parts.append(f"Interaction penalty: -{interaction_penalty:.2f}")
    feedback_parts.append(f"Step reward: {step_reward:.2f}")

    done = step >= MAX_STEPS
    return step_reward, "\n".join(feedback_parts), updated_prescription, done
