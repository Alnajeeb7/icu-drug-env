"""
Quick smoke test — run this before pushing to GitHub.
Tests all 3 tasks with dummy actions to verify no import errors.

Usage:
    python test_local.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_task_easy():
    print("Testing Task 1: single_dose_calc ...", end=" ")
    from env.environment import ICUDrugEnv
    env = ICUDrugEnv(task_name="single_dose_calc", seed=42)
    reset = env.reset()
    assert reset.task_name == "single_dose_calc"
    assert reset.observation.step == 1
    assert reset.observation.patient.weight_kg > 0

    action = {
        "action_type": "prescribe_dose",
        "drug_name": "vancomycin",
        "dose_mg": 1050.0,
        "dose_units": "mg",
        "rationale": "15 mg/kg x 70kg",
    }
    resp = env.step(action)
    assert 0.0 <= resp.reward.value <= 1.0
    env.close()
    print(f"PASS — reward={resp.reward.value:.2f}")


def test_task_medium():
    print("Testing Task 2: interaction_check ...", end=" ")
    from env.environment import ICUDrugEnv
    env = ICUDrugEnv(task_name="interaction_check", seed=42)
    reset = env.reset()
    assert reset.task_name == "interaction_check"

    action = {
        "action_type": "flag_interaction",
        "dangerous_pair": ["warfarin", "ciprofloxacin"],
        "alternative_drug": "amoxicillin",
        "rationale": "Ciprofloxacin inhibits CYP1A2 increasing warfarin levels",
    }
    resp = env.step(action)
    assert 0.0 <= resp.reward.value <= 1.0
    env.close()
    print(f"PASS — reward={resp.reward.value:.2f}")


def test_task_hard():
    print("Testing Task 3: icu_management ...", end=" ")
    from env.environment import ICUDrugEnv
    env = ICUDrugEnv(task_name="icu_management", seed=42)
    reset = env.reset()
    assert reset.task_name == "icu_management"

    rewards = []
    for i in range(10):
        action = {
            "action_type": "adjust_dose",
            "drug_name": "norepinephrine",
            "dose_mg": 0.1,
            "rationale": "Titrating to MAP > 65",
        }
        resp = env.step(action)
        rewards.append(resp.reward.value)
        assert 0.0 <= resp.reward.value <= 1.0
        if resp.done:
            break

    avg = sum(rewards) / len(rewards)
    env.close()
    print(f"PASS — avg_reward={avg:.2f} over {len(rewards)} steps")


def test_state():
    print("Testing state() ...", end=" ")
    from env.environment import ICUDrugEnv
    env = ICUDrugEnv(task_name="single_dose_calc", seed=1)
    env.reset()
    action = {"action_type": "prescribe_dose", "drug_name": "vancomycin", "dose_mg": 500.0}
    env.step(action)
    state = env.state()
    assert state.task_name == "single_dose_calc"
    assert state.step == 1
    assert len(state.reward_history) == 1
    env.close()
    print("PASS")


def test_models():
    print("Testing Pydantic models ...", end=" ")
    from env.models import DrugAction, ICUReward, EnvironmentState
    action = DrugAction(
        action_type="prescribe_dose",
        drug_name="vancomycin",
        dose_mg=1050.0,
        dose_units="mg",
    )
    assert action.drug_name == "vancomycin"
    reward = ICUReward(value=0.85, feedback="Good dose calculation")
    assert 0.0 <= reward.value <= 1.0
    print("PASS")


def test_drug_data():
    print("Testing drug database ...", end=" ")
    from env.drug_data import DRUG_DB, DANGEROUS_INTERACTIONS, VITALS_RANGES
    assert len(DRUG_DB) >= 40, f"Expected 40+ drugs, got {len(DRUG_DB)}"
    assert len(DANGEROUS_INTERACTIONS) >= 30, f"Expected 30+ interactions, got {len(DANGEROUS_INTERACTIONS)}"
    assert "vancomycin" in DRUG_DB
    assert ("warfarin", "aspirin") in DANGEROUS_INTERACTIONS
    print(f"PASS — {len(DRUG_DB)} drugs, {len(DANGEROUS_INTERACTIONS)} interactions")


def test_patient_generator():
    print("Testing patient generator ...", end=" ")
    from env.patient_generator import generate_patient, generate_vitals, evolve_vitals
    patient = generate_patient(seed=99)
    assert patient.weight_kg > 0
    assert patient.age > 0
    vitals = generate_vitals(patient, seed=99, severity="critical")
    assert vitals.map > 0
    evolved = evolve_vitals(vitals, {"norepinephrine": {"dose_mg": 0.1}}, seed=99)
    assert evolved.map > 0
    print("PASS")


if __name__ == "__main__":
    print("=" * 50)
    print("ICU Drug Env — Local Smoke Tests")
    print("=" * 50)
    failed = []
    tests = [
        test_drug_data,
        test_models,
        test_patient_generator,
        test_task_easy,
        test_task_medium,
        test_task_hard,
        test_state,
    ]
    for t in tests:
        try:
            t()
        except Exception as e:
            print(f"FAIL — {e}")
            failed.append(t.__name__)

    print("=" * 50)
    if failed:
        print(f"FAILED: {failed}")
        sys.exit(1)
    else:
        print("ALL TESTS PASSED ✓")
