import logging
from contextlib import asynccontextmanager
from threading import Lock
from typing import Any

import laya
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logger = logging.getLogger("laya_service")
agent: Any | None = None
prediction_lock = Lock()


class PredictionRequest(BaseModel):
    state: dict[str, Any]
    questions: dict[str, dict[str, Any]]


def device_name() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


@asynccontextmanager
async def lifespan(_: FastAPI):
    global agent
    logger.info("loading Laya model on %s", device_name())
    agent = laya.load("convaiinnovations/laya")
    logger.info("Laya model ready")
    try:
        yield
    finally:
        agent = None


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok" if agent is not None else "starting", "device": device_name()}


@app.post("/v1/predict")
def predict(request: PredictionRequest) -> Any:
    if agent is None:
        raise HTTPException(status_code=503, detail="Laya model is still starting")
    try:
        # The local checkpoint is not safe to invoke concurrently. Serialize
        # requests from parallel OMP sessions instead of dropping connections.
        with prediction_lock:
            return agent.predict(request.state, request.questions)
    except Exception as error:
        logger.exception("prediction failed")
        raise HTTPException(status_code=500, detail="Laya prediction failed") from error
