"""Standalone grader for single_dose_calc task."""
from env.tasks.task_easy import grade_action, get_correct_dose

def grade(action, scenario, patient):
    score, feedback = grade_action(action, scenario, patient)
    return {"score": score, "feedback": feedback}
