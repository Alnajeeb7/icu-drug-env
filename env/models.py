"""
Pydantic models for ICU Drug Dosing OpenEnv environment.
Typed Action, Observation, and Reward per OpenEnv spec.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DrugAction(BaseModel):
    """Action model — what the agent submits per step."""
    action_type: str = Field(
        ...,
        description=(
            "Type of action: 'prescribe_dose' | 'flag_interaction' | "
            "'adjust_dose' | 'recommend_alternative' | 'hold_drug'"
        ),
    )
    drug_name: Optional[str] = Field(None, description="Name of drug being acted on")
    dose_mg: Optional[float] = Field(None, description="Proposed dose in milligrams")
    dose_units: Optional[str] = Field(None, description="Units: mg, mcg, units, meq")
    dangerous_pair: Optional[List[str]] = Field(
        None, description="Two-drug pair flagged as dangerous"
    )
    alternative_drug: Optional[str] = Field(None, description="Recommended safe alternative drug")
    rationale: Optional[str] = Field(None, description="Agent's reasoning (used for partial credit)")
    raw_text: Optional[str] = Field(None, description="Raw LLM output before parsing")


class PatientVitals(BaseModel):
    """Current patient vital signs."""
    heart_rate: float = Field(..., description="Heart rate in bpm")
    systolic_bp: float = Field(..., description="Systolic blood pressure mmHg")
    diastolic_bp: float = Field(..., description="Diastolic blood pressure mmHg")
    map: float = Field(..., description="Mean arterial pressure mmHg")
    respiratory_rate: float = Field(..., description="Respiratory rate breaths/min")
    spo2: float = Field(..., description="Oxygen saturation %")
    temperature: float = Field(..., description="Temperature Celsius")
    glucose: float = Field(..., description="Blood glucose mg/dL")
    creatinine: float = Field(..., description="Serum creatinine mg/dL")
    potassium: float = Field(..., description="Serum potassium mEq/L")
    sodium: float = Field(..., description="Serum sodium mEq/L")
    inr: Optional[float] = Field(None, description="INR (if on anticoagulation)")


class PatientProfile(BaseModel):
    """Static patient demographics and clinical context."""
    patient_id: str
    age: int
    weight_kg: float
    sex: str
    diagnosis: str
    comorbidities: List[str]
    allergies: List[str]
    renal_function: str = Field(
        ..., description="normal | mild_impairment | moderate_impairment | severe_impairment | dialysis"
    )
    hepatic_function: str = Field(..., description="normal | mild | moderate | severe")
    current_medications: List[str]


class ICUObservation(BaseModel):
    """Observation returned to the agent after each step."""
    task_name: str
    step: int
    patient: PatientProfile
    vitals: PatientVitals
    available_drugs: List[str]
    current_prescription: Dict[str, Any]
    alert_flags: List[str] = Field(default_factory=list, description="Clinical alert messages")
    lab_results: Dict[str, float] = Field(default_factory=dict)
    task_instruction: str
    previous_action_feedback: Optional[str] = None
    done: bool = False


class ICUReward(BaseModel):
    """Reward signal per step."""
    value: float = Field(..., ge=0.0, le=1.0, description="Reward value 0.0 to 1.0")
    components: Dict[str, float] = Field(
        default_factory=dict,
        description="Breakdown of reward components",
    )
    feedback: str = Field(..., description="Human-readable explanation of reward")
    is_terminal: bool = False
    episode_score: Optional[float] = Field(None, description="Final episode score if terminal")


class EnvironmentState(BaseModel):
    """Full internal state snapshot — returned by state()."""
    task_name: str
    episode_id: str
    step: int
    max_steps: int
    patient: PatientProfile
    vitals: PatientVitals
    prescription_history: List[Dict[str, Any]]
    reward_history: List[float]
    cumulative_reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


class ResetResponse(BaseModel):
    """Response from reset() call."""
    observation: ICUObservation
    task_name: str
    episode_id: str


class StepResponse(BaseModel):
    """Response from step() call."""
    observation: ICUObservation
    reward: ICUReward
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)
