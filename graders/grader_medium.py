"""Standalone grader for interaction_check task."""
from env.tasks.task_medium import grade_action


def grade(action, scenario, step=1):
    score, feedback, done = grade_action(action, scenario, step=step)
    return {"score": score, "feedback": feedback, "done": done}
