"""
Task 2 (Medium): Drug Interaction Detection
Agent receives a patient on 3 drugs, must identify the dangerous pair
and recommend a safe alternative.
Score: 0.5 for correct pair + 0.5 for valid alternative.
"""

import random
from typing import Dict, Any, Tuple, List

from env.drug_data import DANGEROUS_INTERACTIONS, TASK_DRUG_SCENARIOS
from env.models import PatientProfile, PatientVitals, ICUObservation
from env.patient_generator import generate_patient, generate_vitals

TASK_NAME = "interaction_check"
MAX_STEPS = 5


def get_scenario(seed: int = None) -> Dict[str, Any]:
    rng = random.Random(seed)
    return rng.choice(TASK_DRUG_SCENARIOS["medium"])


def build_observation(
    patient: PatientProfile,
    vitals: PatientVitals,
    scenario: Dict[str, Any],
    step: int,
    feedback: str = None,
) -> ICUObservation:
    drugs_str = ", ".join(scenario["current_drugs"])
    alerts = [
        f"PHARMACIST ALERT: Potential interaction detected in current prescription.",
        f"Current medications: {drugs_str}",
        f"Patient context: {scenario['patient']}",
    ]

    instruction = (
        f"TASK: Identify the dangerous drug interaction in this patient's regimen.\n"
        f"Patient: {scenario['patient']}\n"
        f"Current drugs: {drugs_str}\n\n"
        f"One pair of drugs has a MAJOR or CONTRAINDICATED interaction.\n"
        f"Respond with:\n"
        f"  action_type='flag_interaction'\n"
        f"  dangerous_pair=[<drug1>, <drug2>]\n"
        f"  alternative_drug=<safe_replacement>\n"
        f"  rationale=<your reasoning>\n\n"
        f"You have {MAX_STEPS - step + 1} attempts remaining."
    )

    return ICUObservation(
        task_name=TASK_NAME,
        step=step,
        patient=patient,
        vitals=vitals,
        available_drugs=scenario["current_drugs"] + ["amoxicillin", "anidulafungin",
            "pantoprazole", "quetiapine", "insulin_regular", "levetiracetam",
            "rosuvastatin", "labetalol", "acetaminophen"],
        current_prescription={d: {"status": "active"} for d in scenario["current_drugs"]},
        alert_flags=alerts,
        lab_results={
            "creatinine": vitals.creatinine,
            "inr": vitals.inr or 1.0,
            "qt_interval_ms": 440 if "amiodarone" in scenario["current_drugs"] else 390,
        },
        task_instruction=instruction,
        previous_action_feedback=feedback,
    )


def normalize_drug_name(name: str) -> str:
    return name.lower().strip().replace(" ", "_").replace("-", "_")


def check_pair_match(
    proposed_pair: List[str],
    correct_pair: Tuple[str, str],
) -> bool:
    if not proposed_pair or len(proposed_pair) < 2:
        return False
    norm_proposed = {normalize_drug_name(d) for d in proposed_pair}
    norm_correct = {normalize_drug_name(d) for d in correct_pair}
    return norm_proposed == norm_correct


def check_alternative_valid(
    proposed_alternative: str,
    scenario: Dict[str, Any],
) -> bool:
    if not proposed_alternative:
        return False

    norm_alt = normalize_drug_name(proposed_alternative)
    correct_alt = normalize_drug_name(scenario.get("safe_alternative", ""))

    if norm_alt == correct_alt:
        return True

    dangerous_drug_1 = normalize_drug_name(scenario["dangerous_pair"][0])
    dangerous_drug_2 = normalize_drug_name(scenario["dangerous_pair"][1])

    if norm_alt in {dangerous_drug_1, dangerous_drug_2}:
        return False

    safer_alternatives = {
        "acetaminophen", "pantoprazole", "anidulafungin", "levetiracetam",
        "rosuvastatin", "insulin_regular", "quetiapine", "labetalol",
        "amoxicillin", "tinidazole", "meropenem", "pravastatin",
        "reduce_digoxin_by_50_percent",
    }
    if norm_alt in safer_alternatives:
        return True

    return len(norm_alt) > 3


def grade_action(
    action: Dict[str, Any],
    scenario: Dict[str, Any],
    step: int,
) -> Tuple[float, str, bool]:
    """
    Returns (score, feedback, done).
    Score breakdown: 0.5 for correct pair + 0.5 for valid alternative.
    Step penalty applied for late detection.
    """
    proposed_pair = action.get("dangerous_pair", [])
    proposed_alt = action.get("alternative_drug", "")

    pair_correct = check_pair_match(proposed_pair, scenario["dangerous_pair"])
    alt_valid = check_alternative_valid(proposed_alt, scenario)

    pair_score = 0.5 if pair_correct else 0.0
    alt_score = 0.5 if alt_valid else 0.0

    raw_score = pair_score + alt_score
    # Guarantee score is strictly between 0 and 1
    score = round(max(0.0001, raw_score - step_penalty), 4)
    score = max(0.0001, min(0.9999, score))

    feedback_parts = []
    correct_pair_str = " + ".join(scenario["dangerous_pair"])

    if pair_correct:
        feedback_parts.append(f"CORRECT: You identified the dangerous pair ({correct_pair_str}).")
    else:
        proposed_str = " + ".join(proposed_pair) if proposed_pair else "none provided"
        feedback_parts.append(
            f"INCORRECT: You identified '{proposed_str}'. "
            f"The dangerous pair was: {correct_pair_str}."
        )

    if alt_valid:
        feedback_parts.append(f"GOOD: '{proposed_alt}' is an acceptable alternative.")
    else:
        feedback_parts.append(
            f"POOR ALTERNATIVE: '{proposed_alt}'. "
            f"Suggested: {scenario.get('safe_alternative', 'see clinical guidelines')}."
        )

    if step > 1:
        feedback_parts.append(f"Step penalty applied: -{step_penalty:.2f}")

    done = pair_correct and alt_valid
    feedback = " | ".join(feedback_parts)

    return score, feedback, done
