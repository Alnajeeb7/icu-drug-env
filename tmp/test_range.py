import sys
import os

# Add the project root to sys.path
sys.path.insert(0, r"c:\Users\ABDUL AZEEZ\Downloads\icu-drug-env (1)\icu-drug-env")

from env.tasks import task_easy, task_medium, task_hard

def test_clamping():
    print("Testing Easy Task Clamping...")
    # Mock data for task_easy
    scenario = {"drug": "vancomycin"}
    patient = type('obj', (object,), {'weight_kg': 70})()
    
    # Perfect dose
    action = {"dose_mg": 1050.0} # vancomycin 15mg/kg * 70 = 1050
    score, _ = task_easy.grade_action(action, scenario, patient)
    print(f"  Perfect score: {score}")
    assert 0 < score < 1, f"Score {score} not in (0,1)"
    
    # Total failure
    action = {"dose_mg": 10000.0}
    score, _ = task_easy.grade_action(action, scenario, patient)
    print(f"  Failure score: {score}")
    assert 0 < score < 1, f"Score {score} not in (0,1)"

    print("Testing Medium Task Clamping...")
    # Mock scenario for task_medium
    scenario_med = {"dangerous_pair": ("warfarin", "nsaid"), "safe_alternative": "acetaminophen"}
    
    # Perfect detection step 1
    action_med = {"dangerous_pair": ["warfarin", "nsaid"], "alternative_drug": "acetaminophen"}
    score, _, _ = task_medium.grade_action(action_med, scenario_med, step=1)
    print(f"  Perfect score step 1: {score}")
    assert 0 < score < 1, f"Score {score} not in (0,1)"
    
    # Complete miss
    action_med = {"dangerous_pair": ["none", "none"]}
    score, _, _ = task_medium.grade_action(action_med, scenario_med, step=1)
    print(f"  Fail score: {score}")
    assert 0 < score < 1, f"Score {score} not in (0,1)"

    print("\nAll range tests passed!")

if __name__ == "__main__":
    try:
        test_clamping()
    except Exception as e:
        print(f"TEST FAILED: {e}")
        sys.exit(1)
