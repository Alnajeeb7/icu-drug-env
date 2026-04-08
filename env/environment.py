"""
ICU Drug Dosing Environment — main environment class.
Implements OpenEnv spec: reset() / step() / state().
"""

import uuid
import random
from typing import Any, Dict, Optional

from env.models import (
    DrugAction, ICUObservation, ICUReward,
    EnvironmentState, ResetResponse, StepResponse,
)
from env.patient_generator import generate_patient, generate_vitals, evolve_vitals
from env.tasks import task_easy, task_medium, task_hard

TASKS = {
    "single_dose_calc": task_easy,
    "interaction_check": task_medium,
    "icu_management": task_hard,
}

MAX_STEPS = {
    "single_dose_calc": task_easy.MAX_STEPS,
    "interaction_check": task_medium.MAX_STEPS,
    "icu_management": task_hard.MAX_STEPS,
}


class ICUDrugEnv:
    """
    ICU Drug Dosing OpenEnv environment.

    Tasks:
      - single_dose_calc  (easy)   : Calculate correct drug dose
      - interaction_check (medium) : Detect dangerous drug interaction
      - icu_management    (hard)   : Manage full ICU patient over 10 steps
    """

    def __init__(self, task_name: str = "single_dose_calc", seed: int = None):
        if task_name not in TASKS:
            raise ValueError(f"Unknown task '{task_name}'. Choose from: {list(TASKS.keys())}")
        self.task_name = task_name
        self.seed = seed or random.randint(0, 99999)
        self._episode_id: Optional[str] = None
        self._step: int = 0
        self._done: bool = False
        self._patient = None
        self._vitals = None
        self._scenario = None
        self._prescription: Dict[str, Any] = {}
        self._reward_history = []
        self._step_scores = []
        self._last_feedback: Optional[str] = None

    def reset(self) -> ResetResponse:
        self._episode_id = str(uuid.uuid4())[:12]
        self._step = 0
        self._done = False
        self._reward_history = []
        self._step_scores = []
        self._last_feedback = None

        task_mod = TASKS[self.task_name]
        self._scenario = task_mod.get_scenario(seed=self.seed)

        if self.task_name == "icu_management":
            self._patient = generate_patient(
                seed=self.seed,
                diagnosis=self._scenario["diagnosis"],
                renal_state="moderate_impairment",
            )
            self._vitals = generate_vitals(self._patient, seed=self.seed, severity="critical")
            self._prescription = {d: {"status": "active"} for d in self._scenario["initial_drugs"]}
            obs = task_hard.build_observation(
                self._patient, self._vitals, self._scenario,
                step=1, current_prescription=self._prescription,
            )
        elif self.task_name == "interaction_check":
            self._patient = generate_patient(seed=self.seed)
            self._vitals = generate_vitals(self._patient, seed=self.seed, severity="moderate")
            self._prescription = {}
            obs = task_medium.build_observation(
                self._patient, self._vitals, self._scenario, step=1,
            )
        else:
            self._patient = generate_patient(seed=self.seed)
            self._patient.weight_kg = float(self._scenario.get("weight_kg", self._patient.weight_kg))
            self._vitals = generate_vitals(self._patient, seed=self.seed, severity="mild")
            self._prescription = {}
            obs = task_easy.build_observation(
                self._patient, self._vitals, self._scenario, step=1,
            )

        return ResetResponse(
            observation=obs,
            task_name=self.task_name,
            episode_id=self._episode_id,
        )

    def step(self, action: Dict[str, Any]) -> StepResponse:
        if self._done:
            raise RuntimeError("Episode is done. Call reset() first.")

        self._step += 1
        task_mod = TASKS[self.task_name]
        max_steps = MAX_STEPS[self.task_name]

        if self.task_name == "single_dose_calc":
            score, feedback = task_easy.grade_action(action, self._scenario, self._patient)
            done = self._step >= max_steps or score >= 0.95
            reward = ICUReward(
                value=round(score, 4),
                components={"dose_accuracy": max(0.0001, min(0.9999, score))},
                feedback=feedback,
                is_terminal=done,
                episode_score=score if done else None,
            )
            if not done:
                obs = task_easy.build_observation(
                    self._patient, self._vitals, self._scenario,
                    step=self._step + 1, feedback=feedback,
                )
            else:
                obs = task_easy.build_observation(
                    self._patient, self._vitals, self._scenario,
                    step=self._step, feedback=feedback,
                )
                obs.done = True

        elif self.task_name == "interaction_check":
            score, feedback, done = task_medium.grade_action(
                action, self._scenario, step=self._step
            )
            if self._step >= max_steps:
                done = True
            reward = ICUReward(
                value=round(score, 4),
                components={
                    "pair_identified": 0.4999 if score >= 0.5 else 0.0001,
                    "alternative_valid": 0.4999 if score >= 0.9 else 0.0001,
                },
                feedback=feedback,
                is_terminal=done,
                episode_score=score if done else None,
            )
            obs = task_medium.build_observation(
                self._patient, self._vitals, self._scenario,
                step=self._step + 1 if not done else self._step,
                feedback=feedback,
            )
            obs.done = done

        elif self.task_name == "icu_management":
            step_reward, feedback, updated_prescription, done = task_hard.grade_action(
                action=action,
                vitals=self._vitals,
                prescription=self._prescription,
                scenario=self._scenario,
                step=self._step,
                step_scores=self._step_scores,
            )
            self._prescription = updated_prescription
            self._step_scores.append(step_reward)

            self._vitals = evolve_vitals(
                self._vitals, self._prescription, seed=self.seed + self._step
            )

            episode_score = None
            if done:
                avg_score = sum(self._step_scores) / len(self._step_scores) if self._step_scores else 0.0001
                episode_score = round(max(0.0001, min(0.9999, avg_score)), 4)

            reward = ICUReward(
                value=round(step_reward, 4),
                components={"vitals_score": max(0.0001, min(0.9999, step_reward))},
                feedback=feedback,
                is_terminal=done,
                episode_score=episode_score,
            )
            obs = task_hard.build_observation(
                self._patient, self._vitals, self._scenario,
                step=self._step + 1 if not done else self._step,
                current_prescription=self._prescription,
                feedback=feedback,
            )
            obs.done = done
        else:
            raise ValueError(f"Unknown task: {self.task_name}")

        self._done = done
        self._reward_history.append(reward.value)
        self._last_feedback = feedback

        return StepResponse(
            observation=obs,
            reward=reward,
            done=done,
            info={
                "episode_id": self._episode_id,
                "step": self._step,
                "task": self.task_name,
                "cumulative_reward": max(0.0001, min(0.9999 * self._step, round(sum(self._reward_history), 4))),
            },
        )

    def state(self) -> EnvironmentState:
        return EnvironmentState(
            task_name=self.task_name,
            episode_id=self._episode_id or "",
            step=self._step,
            max_steps=MAX_STEPS[self.task_name],
            patient=self._patient,
            vitals=self._vitals,
            prescription_history=[],
            reward_history=self._reward_history,
            cumulative_reward=round(sum(self._reward_history), 4) if self._reward_history else 0.0,
            done=self._done,
            info={"scenario": str(self._scenario)},
        )

    def close(self):
        self._done = True
