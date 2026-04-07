"""Standalone grader for icu_management task."""
from env.tasks.task_hard import grade_action, score_vitals


def grade(action, vitals, prescription, scenario, step, step_scores):
    step_reward, feedback, updated_prescription, done = grade_action(
        action=action,
        vitals=vitals,
        prescription=prescription,
        scenario=scenario,
        step=step,
        step_scores=step_scores,
    )
    return {
        "step_reward": step_reward,
        "feedback": feedback,
        "updated_prescription": updated_prescription,
        "done": done,
        "episode_score": round(sum(step_scores + [step_reward]) / (len(step_scores) + 1), 4),
    }
