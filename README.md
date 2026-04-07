# ICU Drug Dosing Environment 🏥

An OpenEnv-compatible reinforcement learning environment that simulates real ICU clinical decision-making. AI agents must calculate drug doses, detect dangerous drug interactions, and manage critically ill patients.

**No external data dependencies. Fully self-contained. Runs offline.**

---

## Why This Environment?

Drug errors kill ~7,000 people/year in the US alone. ICU pharmacists juggle dozens of drugs per patient, each with complex interactions, renal adjustments, and dose calculations. This environment trains agents to make these decisions correctly — a genuine clinical skill with life-or-death stakes.

**This is the first medical/pharmacology environment in the OpenEnv ecosystem.**

---

## Tasks

### Task 1: `single_dose_calc` (Easy)
Agent receives a patient (weight, diagnosis, renal function) and a drug name. Must calculate the correct dose in mg.

- **Score**: `1.0 - |proposed - correct| / correct`, clamped 0–1
- **Max steps**: 3
- **Example**: Vancomycin for a 70kg MRSA patient → 1050mg

### Task 2: `interaction_check` (Medium)
Agent receives a patient on 3 concurrent medications. Must identify the dangerous drug-drug interaction pair and recommend a safe alternative.

- **Score**: 0.5 for correct pair + 0.5 for valid alternative − step penalty
- **Max steps**: 5
- **Example**: Warfarin + Ciprofloxacin → INR elevation → switch to Amoxicillin

### Task 3: `icu_management` (Hard)
Agent manages a critically ill patient (septic shock, multi-organ dysfunction) over 10 steps. Vitals evolve each step. Agent must keep MAP, glucose, HR, SpO2, and potassium in target ranges while avoiding dangerous drug combinations.

- **Score**: Rolling average of vitals-in-range − interaction penalties
- **Max steps**: 10
- **Targets**: MAP 65-100, Glucose 140-180, HR 60-100, SpO2 >94%, K+ 3.5-5.0

---

## Action Space

```json
{
  "action_type": "prescribe_dose | flag_interaction | adjust_dose | hold_drug | recommend_alternative",
  "drug_name": "vancomycin",
  "dose_mg": 1050.0,
  "dose_units": "mg",
  "dangerous_pair": ["warfarin", "ciprofloxacin"],
  "alternative_drug": "amoxicillin",
  "rationale": "15 mg/kg for 70kg patient with normal renal function"
}
```

## Observation Space

```json
{
  "task_name": "single_dose_calc",
  "step": 1,
  "patient": { "weight_kg": 70, "age": 65, "diagnosis": "septic shock", "renal_function": "normal" },
  "vitals": { "map": 72, "heart_rate": 95, "spo2": 96, "glucose": 180, "creatinine": 1.1 },
  "available_drugs": ["vancomycin", "piperacillin_tazobactam", ...],
  "current_prescription": {},
  "alert_flags": [],
  "task_instruction": "Calculate the correct dose for this patient..."
}
```

---

## Baseline Scores

| Task | Model | Score |
|------|-------|-------|
| single_dose_calc | Qwen2.5-72B-Instruct | ~0.82 |
| interaction_check | Qwen2.5-72B-Instruct | ~0.71 |
| icu_management | Qwen2.5-72B-Instruct | ~0.54 |

---

## Getting Your HuggingFace Token

You need a free HuggingFace API token to run the inference script (for LLM access via the HF Inference API).

1. Create a free account at [huggingface.co](https://huggingface.co/join)
2. Go to **Settings** → **Access Tokens**: [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
3. Click **New token**
4. Give it a name (e.g. `icu-drug-env`) and select **Read** permission (or **Write** if you also want to push to HF Spaces)
5. Click **Generate** and copy the token (starts with `hf_...`)

Set it as an environment variable before running inference:

```bash
# Linux/macOS
export HF_TOKEN="hf_your_token_here"

# Windows PowerShell
$env:HF_TOKEN = "hf_your_token_here"
```

> ⚠️ **Never commit your token to code or share it publicly.** Use environment variables only.

---

## Setup & Usage

### Local Python
```bash
git clone https://github.com/Alnajeeb7/icu-drug-env
cd icu-drug-env
pip install -r requirements.txt

# Start server
uvicorn server.app:app --host 0.0.0.0 --port 7860

# Run inference
export HF_TOKEN=your_token
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export API_BASE_URL=https://router.huggingface.co/v1
python inference.py
```

### Docker
```bash
docker build -t icu-drug-env .
docker run -p 7860:7860 \
  -e HF_TOKEN=your_token \
  -e MODEL_NAME=Qwen/Qwen2.5-72B-Instruct \
  -e API_BASE_URL=https://router.huggingface.co/v1 \
  icu-drug-env
```

### API Usage
```python
import requests

# Reset environment
resp = requests.post("http://localhost:7860/reset", json={"task_name": "single_dose_calc"})
session_id = resp.json()["session_id"]
obs = resp.json()["observation"]

# Take a step
action = {
    "action_type": "prescribe_dose",
    "drug_name": "vancomycin",
    "dose_mg": 1050.0,
    "rationale": "15 mg/kg × 70kg"
}
resp = requests.post("http://localhost:7860/step", json={"session_id": session_id, "action": action})
print(resp.json()["reward"])
```

### WebSocket Usage
```python
import asyncio, websockets, json

async def run():
    async with websockets.connect("ws://localhost:7860/ws") as ws:
        await ws.send(json.dumps({"command": "reset", "task_name": "icu_management"}))
        msg = json.loads(await ws.recv())
        print(msg["observation"]["task_instruction"])

asyncio.run(run())
```

---

## Drug Database

Contains 50+ real ICU drugs and 60+ documented dangerous interactions including:
- Warfarin + Ciprofloxacin → INR elevation
- Digoxin + Amiodarone → Digoxin toxicity
- Vancomycin + Pip-Tazo → Nephrotoxicity
- SSRI + MAOI → Serotonin syndrome (contraindicated)
- Fentanyl + Midazolam → Respiratory depression

All interaction data sourced from public clinical pharmacology knowledge (no proprietary databases).

---

## Environment Design

- **Deterministic**: Same seed = same episode (reproducible baselines)
- **Dense rewards**: Every step provides signal, not just episode end
- **Partial credit**: Wrong but close answers score > 0
- **Realistic vitals evolution**: Drugs affect vitals realistically each step
- **No external dependencies**: Runs fully offline inside Docker

---

## Citation

```bibtex
@software{icu_drug_env,
  title = {ICU Drug Dosing OpenEnv Environment},
  year = {2025},
  note = {OpenEnv-compatible clinical pharmacology RL environment}
}
```
