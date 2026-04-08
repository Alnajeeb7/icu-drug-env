"""
Task 1 (Easy): Single Drug Dose Calculation
Agent receives patient weight + drug + condition.
Agent must output the correct dose in mg.
"""

import random
from typing import Dict, Any, Tuple

from env.drug_data import DRUG_DB, TASK_DRUG_SCENARIOS, DOSE_REFERENCE
from env.models import PatientProfile, PatientVitals, ICUObservation
from env.patient_generator import generate_patient, generate_vitals

TASK_NAME = "single_dose_calc"
MAX_STEPS = 3


def get_scenario(seed: int = None) -> Dict[str, Any]:
    rng = random.Random(seed)
    scenarios = TASK_DRUG_SCENARIOS["easy"]
    return rng.choice(scenarios)


def get_correct_dose(scenario: Dict[str, Any], weight_kg: float) -> float:
    drug = scenario["drug"]
    db = DRUG_DB.get(drug, {})

    if drug == "vancomycin":
        return round(db["typical_dose_mg_kg"] * weight_kg, 1)
    elif drug == "furosemide":
        return round(db["typical_dose_mg_kg"] * weight_kg, 1)
    elif drug == "acetaminophen":
        dose = db["typical_dose_mg_kg"] * weight_kg
        return round(min(dose, 1000.0), 1)
    elif drug == "dexamethasone":
        return 6.0
    elif drug == "ondansetron":
        dose = db["typical_dose_mg_kg"] * weight_kg
        return round(min(dose, 8.0), 1)
    elif drug == "morphine":
        return round(db["typical_dose_mg_kg"] * weight_kg, 1)
    else:
        if "fixed_dose_mg" in db:
            return db["fixed_dose_mg"]
        return round(db.get("typical_dose_mg_kg", 10.0) * weight_kg, 1)


def build_observation(
    patient: PatientProfile,
    vitals: PatientVitals,
    scenario: Dict[str, Any],
    step: int,
    feedback: str = None,
) -> ICUObservation:
    drug = scenario["drug"]
    condition = scenario["condition"]
    db = DRUG_DB.get(drug, {})

    instruction = (
        f"TASK: Calculate the correct dose for this patient.\n"
        f"Drug: {drug.replace('_', ' ').title()}\n"
        f"Indication: {condition}\n"
        f"Patient weight: {patient.weight_kg} kg\n"
        f"Renal function: {patient.renal_function}\n"
        f"Drug class: {db.get('class', 'unknown')}\n\n"
        f"Respond with action_type='prescribe_dose', drug_name='{drug}', "
        f"dose_mg=<your_calculated_dose>, dose_units='mg'.\n"
        f"Provide your rationale in the rationale field."
    )

    return ICUObservation(
        task_name=TASK_NAME,
        step=step,
        patient=patient,
        vitals=vitals,
        available_drugs=list(DRUG_DB.keys()),
        current_prescription={},
        alert_flags=[],
        lab_results={
            "creatinine": vitals.creatinine,
            "glucose": vitals.glucose,
            "potassium": vitals.potassium,
        },
        task_instruction=instruction,
        previous_action_feedback=feedback,
    )


def grade_action(
    action: Dict[str, Any],
    scenario: Dict[str, Any],
    patient: PatientProfile,
) -> Tuple[float, str]:
    """
    Deterministic grader. Score = 1.0 - |proposed - correct| / correct, clamped 0-1.
    Partial credit for being within range.
    """
    correct_dose = get_correct_dose(scenario, patient.weight_kg)
    proposed_dose = action.get("dose_mg")

    if proposed_dose is None:
        return 0.0001, f"No dose provided. Correct dose was {correct_dose} mg."

    try:
        proposed_dose = float(proposed_dose)
    except (ValueError, TypeError):
        return 0.0001, f"Invalid dose format. Correct dose was {correct_dose} mg."

    if correct_dose == 0:
        return 0.0001, "Reference dose is 0 — scenario error."

    error_ratio = abs(proposed_dose - correct_dose) / correct_dose
    # Guarantee score is strictly between 0 and 1
    score = round(max(0.0001, 1.0 - error_ratio), 4)
    score = max(0.0001, min(0.9999, score))

    if score >= 0.95:
        feedback = f"Excellent! Dose {proposed_dose} mg is correct (target: {correct_dose} mg)."
    elif score >= 0.75:
        feedback = f"Good. Dose {proposed_dose} mg is within 25% of target {correct_dose} mg. Score: {score:.2f}"
    elif score >= 0.5:
        feedback = f"Partial credit. Dose {proposed_dose} mg — target was {correct_dose} mg. Score: {score:.2f}"
    else:
        feedback = f"Incorrect. Proposed {proposed_dose} mg, correct was {correct_dose} mg. Score: {score:.2f}"

    return score, feedback
