import json
import logging
import time
from contextlib import asynccontextmanager
from threading import Lock
from typing import Any, Literal

import laya
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logger = logging.getLogger("laya_service")
agent: Any | None = None
prediction_lock = Lock()
MODEL_ID = "laya"
CHECKPOINT_ID = "convaiinnovations/laya-typed-decisions"


class PredictionRequest(BaseModel):
    state: dict[str, Any]
    questions: dict[str, dict[str, Any]]


class SystemOneRequest(BaseModel):
    model: str | None = None
    state: Any
    questions: dict[str, dict[str, Any]]


class ChatMessage(BaseModel):
    role: Literal["system", "developer", "user", "assistant"]
    content: str | list[dict[str, Any]] | None = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    stream: bool = False
    tools: list[dict[str, Any]] | None = None
    stream_options: dict[str, Any] | None = None


def device_name() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


@asynccontextmanager
async def lifespan(_: FastAPI):
    global agent
    logger.info("loading Laya checkpoint %s on %s", CHECKPOINT_ID, device_name())
    agent = laya.load(CHECKPOINT_ID)
    logger.info("Laya checkpoint ready")
    try:
        yield
    finally:
        agent = None


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok" if agent is not None else "starting", "device": device_name()}


@app.get("/v1/models")
def models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [{"id": MODEL_ID, "object": "model", "created": 0, "owned_by": "omp-laya"}],
    }


def message_text(content: str | list[dict[str, Any]] | None) -> str:
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    return "\n".join(
        part["text"] for part in content if part.get("type") == "text" and isinstance(part.get("text"), str)
    )


DEFAULT_JUDGE_INSTRUCTIONS = (
    "Judge the assistant response in the supplied conversation for correctness, relevance, and completeness."
)
JUDGE_CRITERIA = {
    "0": "Incorrect or fails the request.",
    "1": "Major problems; only a small part is correct.",
    "2": "Mostly correct; minor omissions or issues.",
    "3": "Fully correct, relevant, and complete.",
}
CORRECTNESS_CRITERIA = {
    "true": "The answer is correct and fully satisfies the request.",
    "false": "The answer is wrong, irrelevant, or incomplete.",
}
CORRECTNESS_THRESHOLD = 0.6
CORRECTNESS_INSTRUCTIONS = (
    "Is the answer factually correct and does it satisfy every explicit requirement? "
    "Answer true only if both conditions hold, otherwise false."
)


def judge_inputs(messages: list[ChatMessage]) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    rubric = "\n\n".join(
        text for message in messages
        if message.role in ("system", "developer")
        if (text := message_text(message.content))
    )
    instructions = rubric or DEFAULT_JUDGE_INSTRUCTIONS
    state = "\n\n".join(
        f"{message.role}: {text}" for message in messages
        if message.role not in ("system", "developer")
        if (text := message_text(message.content))
    )
    if not state:
        state = DEFAULT_JUDGE_INSTRUCTIONS
    questions = {
        "judge": {
            "type": "choice",
            "instructions": instructions,
            "criteria": JUDGE_CRITERIA,
        },
        "correctness": {
            "type": "noul",
            "instructions": CORRECTNESS_INSTRUCTIONS,
            "criteria": CORRECTNESS_CRITERIA,
        },
    }
    return {"input": state}, questions

def completion_response(content: str, prompt_tokens: int, completion_tokens: int) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-laya-{int(time.time() * 1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_ID,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def completion_stream(content: str, prompt_tokens: int, completion_tokens: int, include_usage: bool):
    completion_id = f"chatcmpl-laya-{int(time.time() * 1000)}"
    created = int(time.time())
    base = {"id": completion_id, "object": "chat.completion.chunk", "created": created, "model": MODEL_ID}
    chunks = [
        {**base, "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]},
        {**base, "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}]},
        {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    ]
    if include_usage:
        chunks.append({
            **base,
            "choices": [],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        })
    for chunk in chunks:
        yield f"data: {json.dumps(chunk)}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest) -> Any:
    if request.model != MODEL_ID:
        raise HTTPException(status_code=404, detail=f"Unknown Laya model: {request.model}")
    if not request.messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")
    if agent is None:
        raise HTTPException(status_code=503, detail="Laya model is still starting")

    state, questions = judge_inputs(request.messages)
    try:
        with prediction_lock:
            prediction = agent.predict(state, questions)
    except Exception as error:
        logger.exception("judge prediction failed")
        raise HTTPException(status_code=500, detail="Laya judge prediction failed") from error

    try:
        judge = prediction["answers"]["judge"]
        correctness = prediction["answers"]["correctness"]
        raw_score = int(judge["choice"])
        if str(raw_score) not in JUDGE_CRITERIA:
            raise ValueError("Laya returned unknown judge label")
        correctness_probability = float(correctness["noul"])
        score = raw_score if correctness_probability >= CORRECTNESS_THRESHOLD else min(raw_score, 1)
        confidence = min(float(judge["confidence"]), float(correctness["confidence"]))
        probabilities = judge["probabilities"]
        prompt_tokens = int(prediction["usage"]["input_tokens"])
    except (KeyError, TypeError, ValueError) as error:
        logger.exception("Laya returned an invalid judge prediction")
        raise HTTPException(status_code=502, detail="Laya returned an invalid judge prediction") from error

    content = json.dumps(
        {
            "score": score,
            "raw_score": raw_score,
            "confidence": round(confidence, 4),
            "correctness_probability": round(correctness_probability, 4),
            "probabilities": probabilities,
        },
        separators=(",", ":"),
    )
    completion_tokens = max(1, (len(content) + 3) // 4)
    if request.stream:
        return StreamingResponse(
            completion_stream(
                content,
                prompt_tokens,
                completion_tokens,
                bool(request.stream_options and request.stream_options.get("include_usage")),
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )
    return completion_response(content, prompt_tokens, completion_tokens)


@app.post("/v1/predict")
def predict(request: PredictionRequest) -> Any:
    if agent is None:
        raise HTTPException(status_code=503, detail="Laya model is still starting")
    try:
        with prediction_lock:
            return agent.predict(request.state, request.questions)
    except Exception as error:
        logger.exception("prediction failed")
        raise HTTPException(status_code=500, detail="Laya prediction failed") from error


@app.post("/v1/systemone")
def system_one(request: SystemOneRequest) -> Any:
    if agent is None:
        raise HTTPException(status_code=503, detail="Laya model is still starting")
    try:
        # This endpoint speaks the native System One / Jev request and response
        # contract. The loaded checkpoint is fixed by this extension's config.
        with prediction_lock:
            return agent.predict(request.state, request.questions)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        logger.exception("System One prediction failed")
        raise HTTPException(status_code=500, detail="Laya System One prediction failed") from error
