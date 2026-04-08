"""
FastAPI server exposing OpenEnv API:
  POST /reset
  POST /step
  GET  /state
  GET  /health
  GET  /tasks
WebSocket /ws — for real-time step-by-step interaction
"""

import json
import uuid
from typing import Any, Dict, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from env.environment import ICUDrugEnv

app = FastAPI(
    title="ICU Drug Dosing OpenEnv",
    description="OpenEnv-compatible ICU drug dosing and interaction environment.",
    version="1.0.0",
)

_sessions: Dict[str, ICUDrugEnv] = {}


class ResetRequest(BaseModel):
    task_name: str = "single_dose_calc"
    seed: Optional[int] = None
    session_id: Optional[str] = None


class StepRequest(BaseModel):
    session_id: str
    action: Dict[str, Any]


@app.get("/")
def root():
    return {
        "env": "ICU Drug Dosing Environment",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "POST /reset": "Start a new episode",
            "POST /step": "Take an action",
            "GET /state": "Get current state",
            "GET /health": "Health check",
            "GET /tasks": "List available tasks",
            "WS /ws": "WebSocket interface",
        },
    }


@app.get("/health")
def health():
    return {"status": "ok", "env": "icu-drug-env", "version": "1.0.0"}


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {
                "name": "single_dose_calc",
                "difficulty": "easy",
                "description": "Calculate the correct drug dose for a patient given weight and condition.",
                "max_steps": 3,
                "score_range": [0.0, 1.0],
            },
            {
                "name": "interaction_check",
                "difficulty": "medium",
                "description": "Identify dangerous drug-drug interactions and recommend safe alternatives.",
                "max_steps": 5,
                "score_range": [0.0, 1.0],
            },
            {
                "name": "icu_management",
                "difficulty": "hard",
                "description": "Manage a critically ill ICU patient with evolving vitals over 10 steps.",
                "max_steps": 10,
                "score_range": [0.0, 1.0],
            },
        ]
    }


@app.post("/reset")
def reset(request: ResetRequest = Body(default=None)):
    if request is None:
        request = ResetRequest()
    session_id = request.session_id or str(uuid.uuid4())[:12]
    env = ICUDrugEnv(task_name=request.task_name, seed=request.seed)
    _sessions[session_id] = env
    response = env.reset()
    return {
        "session_id": session_id,
        "task_name": response.task_name,
        "episode_id": response.episode_id,
        "observation": response.observation.model_dump(),
    }


@app.post("/step")
def step(request: StepRequest):
    env = _sessions.get(request.session_id)
    if not env:
        raise HTTPException(status_code=404, detail=f"Session '{request.session_id}' not found. Call /reset first.")
    if env._done:
        raise HTTPException(status_code=400, detail="Episode is done. Call /reset to start a new episode.")
    try:
        response = env.step(request.action)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "observation": response.observation.model_dump(),
        "reward": response.reward.model_dump(),
        "done": response.done,
        "info": response.info,
    }


@app.get("/state")
def state(session_id: str):
    env = _sessions.get(session_id)
    if not env:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return env.state().model_dump()


@app.delete("/session/{session_id}")
def close_session(session_id: str):
    env = _sessions.pop(session_id, None)
    if env:
        env.close()
    return {"status": "closed", "session_id": session_id}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    env: Optional[ICUDrugEnv] = None
    session_id = str(uuid.uuid4())[:12]

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"error": "Invalid JSON"})
                continue

            cmd = msg.get("command", "")

            if cmd == "reset":
                task_name = msg.get("task_name", "single_dose_calc")
                seed = msg.get("seed", None)
                env = ICUDrugEnv(task_name=task_name, seed=seed)
                _sessions[session_id] = env
                response = env.reset()
                await websocket.send_json({
                    "type": "reset",
                    "session_id": session_id,
                    "task_name": response.task_name,
                    "episode_id": response.episode_id,
                    "observation": response.observation.model_dump(),
                })

            elif cmd == "step":
                if not env:
                    await websocket.send_json({"error": "Call reset first"})
                    continue
                action = msg.get("action", {})
                try:
                    response = env.step(action)
                    await websocket.send_json({
                        "type": "step",
                        "observation": response.observation.model_dump(),
                        "reward": response.reward.model_dump(),
                        "done": response.done,
                        "info": response.info,
                    })
                except RuntimeError as e:
                    await websocket.send_json({"error": str(e)})

            elif cmd == "state":
                if not env:
                    await websocket.send_json({"error": "No active session"})
                    continue
                await websocket.send_json({
                    "type": "state",
                    "state": env.state().model_dump(),
                })

            elif cmd == "close":
                if env:
                    env.close()
                await websocket.send_json({"type": "closed"})
                break

            else:
                await websocket.send_json({"error": f"Unknown command: {cmd}"})

    except WebSocketDisconnect:
        if env:
            env.close()
        _sessions.pop(session_id, None)
