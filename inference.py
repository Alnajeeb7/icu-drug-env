"""
Inference Script — ICU Drug Dosing OpenEnv
==========================================
Runs an LLM agent against all 3 tasks and emits structured stdout logs.

Required env vars:
  API_BASE_URL  - LLM endpoint (default: HuggingFace router)
  MODEL_NAME    - Model to use
  HF_TOKEN      - API key

Stdout format (MANDATORY):
  [START] task=<name> env=icu-drug-env model=<model>
  [STEP]  step=<n> action=<str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...>
"""

import json
import os
import sys
import traceback
from typing import Any, Dict, List, Optional

from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env.environment import ICUDrugEnv

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")  # Optional — if using from_docker_image()

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

BENCHMARK = "icu-drug-env"
MAX_STEPS_OVERRIDE = None
TEMPERATURE = 0.3
SEED = 42

TASKS = ["single_dose_calc", "interaction_check", "icu_management"]

SYSTEM_PROMPT = """You are an expert ICU clinical pharmacist AI agent.
You interact with a simulated ICU environment by taking structured actions.
Always respond with ONLY valid JSON in this exact format:

{
  "action_type": "prescribe_dose" | "flag_interaction" | "adjust_dose" | "hold_drug" | "recommend_alternative",
  "drug_name": "<drug name or null>",
  "dose_mg": <number or null>,
  "dose_units": "mg" | "mcg" | "units" | "meq" | null,
  "dangerous_pair": ["<drug1>", "<drug2>"] or null,
  "alternative_drug": "<drug name or null>",
  "rationale": "<your clinical reasoning>"
}

Rules:
- drug names: use snake_case (e.g. vancomycin, piperacillin_tazobactam, insulin_regular)
- dose_mg: provide numeric value only, no units in the number field
- dangerous_pair: exactly 2 drug names when flagging interaction
- rationale: brief clinical justification
- ONLY return JSON — no other text
"""


def build_user_prompt(observation: Dict[str, Any]) -> str:
    return (
        f"Task: {observation.get('task_name')}\n"
        f"Step: {observation.get('step')}\n\n"
        f"INSTRUCTION:\n{observation.get('task_instruction', '')}\n\n"
        f"PATIENT:\n"
        f"  Weight: {observation['patient']['weight_kg']} kg\n"
        f"  Age: {observation['patient']['age']}\n"
        f"  Diagnosis: {observation['patient']['diagnosis']}\n"
        f"  Renal function: {observation['patient']['renal_function']}\n"
        f"  Allergies: {observation['patient']['allergies']}\n\n"
        f"VITALS:\n"
        f"  MAP: {observation['vitals']['map']} mmHg\n"
        f"  HR: {observation['vitals']['heart_rate']} bpm\n"
        f"  SpO2: {observation['vitals']['spo2']}%\n"
        f"  Glucose: {observation['vitals']['glucose']} mg/dL\n"
        f"  Creatinine: {observation['vitals']['creatinine']} mg/dL\n"
        f"  Potassium: {observation['vitals']['potassium']} mEq/L\n\n"
        f"ALERTS: {'; '.join(observation.get('alert_flags', [])) or 'None'}\n\n"
        f"CURRENT PRESCRIPTION: {list(observation.get('current_prescription', {}).keys()) or 'None'}\n\n"
        f"PREVIOUS FEEDBACK: {observation.get('previous_action_feedback') or 'None'}\n\n"
        f"Respond with ONLY JSON action."
    )


def parse_llm_action(raw_text: str) -> Dict[str, Any]:
    raw_text = raw_text.strip()
    if raw_text.startswith("```"):
        lines = raw_text.split("\n")
        raw_text = "\n".join(lines[1:-1]) if len(lines) > 2 else raw_text
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        start = raw_text.find("{")
        end = raw_text.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(raw_text[start:end])
            except json.JSONDecodeError:
                pass
    return {
        "action_type": "prescribe_dose",
        "drug_name": None,
        "dose_mg": None,
        "rationale": raw_text[:200],
        "raw_text": raw_text,
    }


def run_episode(
    client: OpenAI,
    task_name: str,
    seed: int = SEED,
) -> Dict[str, Any]:
    env = ICUDrugEnv(task_name=task_name, seed=seed)
    reset_response = env.reset()
    obs = reset_response.observation.model_dump()

    print(f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}", flush=True)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    rewards: List[float] = []
    steps_taken = 0
    final_score = 0.0
    success = False
    last_error = None

    max_steps_for_task = {"single_dose_calc": 3, "interaction_check": 5, "icu_management": 10}
    max_steps = max_steps_for_task.get(task_name, 10)

    done = False
    while not done and steps_taken < max_steps:
        user_msg = build_user_prompt(obs)
        messages.append({"role": "user", "content": user_msg})

        raw_action = ""
        error_str = "null"
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=512,
            )
            raw_action = response.choices[0].message.content or ""
            action = parse_llm_action(raw_action)
            messages.append({"role": "assistant", "content": raw_action})
        except Exception as e:
            error_str = str(e)[:100].replace("\n", " ")
            action = {"action_type": "prescribe_dose", "drug_name": None, "dose_mg": None}
            last_error = error_str

        step_response = env.step(action)
        reward_val = step_response.reward.value
        done = step_response.done
        obs = step_response.observation.model_dump()
        steps_taken += 1
        rewards.append(reward_val)

        action_str = (
            f"{action.get('action_type','?')}("
            f"{action.get('drug_name','?')},"
            f"dose={action.get('dose_mg','?')},"
            f"pair={action.get('dangerous_pair','?')})"
        ).replace(" ", "").replace("'", "")

        print(
            f"[STEP] step={steps_taken} action={action_str} "
            f"reward={reward_val:.2f} done={'true' if done else 'false'} "
            f"error={error_str}",
            flush=True,
        )

        if step_response.reward.episode_score is not None:
            final_score = step_response.reward.episode_score

    if not rewards:
        rewards = [0.0]
    if final_score == 0.0:
        final_score = rewards[-1] if task_name != "icu_management" else round(sum(rewards) / len(rewards), 2)

    success = final_score >= 0.5
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)

    env.close()

    print(
        f"[END] success={'true' if success else 'false'} steps={steps_taken} "
        f"rewards={rewards_str}",
        flush=True,
    )

    return {
        "task": task_name,
        "score": final_score,
        "success": success,
        "steps": steps_taken,
        "rewards": rewards,
    }


def main():
    client = OpenAI(
        api_key=HF_TOKEN,
        base_url=API_BASE_URL,
    )

    results = []
    for task_name in TASKS:
        try:
            result = run_episode(client, task_name, seed=SEED)
            results.append(result)
        except Exception as e:
            print(f"ERROR running task {task_name}: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            print(
                f"[END] success=false steps=0 rewards=0.00",
                flush=True,
            )
            results.append({"task": task_name, "score": 0.0, "success": False})

    # Summary goes to stderr to avoid polluting structured stdout output
    total = 0.0
    print("\n===== FINAL RESULTS =====", file=sys.stderr)
    for r in results:
        print(f"  {r['task']}: score={r.get('score', 0.0):.2f} success={r.get('success', False)}", file=sys.stderr)
        total += r.get("score", 0.0)
    avg = total / len(results) if results else 0.0
    print(f"  AVERAGE SCORE: {avg:.2f}", file=sys.stderr)


if __name__ == "__main__":
    main()
