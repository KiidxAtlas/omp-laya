import json
import time
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import laya_server


class FakeAgent:
    def __init__(self):
        self.state = None
        self.questions = None
        self.correctness_probability = 0.8

    def predict(self, state, questions):
        self.state = state
        self.questions = questions
        assert questions["judge"]["type"] == "choice"
        assert questions["correctness"]["type"] == "noul"
        return {
            "answers": {
                "judge": {
                    "choice": "2",
                    "confidence": 0.91,
                    "probabilities": {"0": 0.01, "1": 0.02, "2": 0.18, "3": 0.79},
                },
                "correctness": {
                    "noul": self.correctness_probability,
                    "confidence": max(self.correctness_probability, 1 - self.correctness_probability),
                },
            },
            "usage": {"input_tokens": 13, "output_tokens": 0},
        }

    def predict_batch(self, states, questions):
        return [
            {
                "model": "laya-rl-agent",
                "answers": {qid: {"type": "noul", "noul": 0.9 if "rm" in state else 0.1} for qid in questions},
                "usage": {"input_tokens": len(state), "output_tokens": 0},
            }
            for state in states
        ]


class LayaOpenAICompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.fake_agent = FakeAgent()
        self.agent_patch = patch.object(laya_server, "agent", self.fake_agent)
        self.agent_patch.start()
        self.client = TestClient(laya_server.app)

    def tearDown(self):
        self.agent_patch.stop()

    def test_model_list_exposes_laya(self):
        response = self.client.get("/v1/models")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["data"][0]["id"], "laya")


    def test_systemone_endpoint_preserves_jev_wire_contract(self):
        state = [{"role": "user", "content": "Evaluate this run"}]
        questions = {
            "judge": {"type": "choice", "instructions": "Rate the run", "criteria": {"0": "bad", "3": "good"}},
            "correctness": {"type": "noul", "instructions": "Is it correct?"},
        }
        response = self.client.post(
            "/v1/systemone",
            json={"model": "jev-1.13.0", "state": state, "questions": questions},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.fake_agent.state, state)
        self.assertEqual(self.fake_agent.questions, questions)
        self.assertEqual(response.json()["answers"]["judge"]["choice"], "2")
        self.assertEqual(response.json()["usage"]["input_tokens"], 13)
    def test_chat_completion_returns_structured_score(self):
        response = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "laya",
                "messages": [
                    {"role": "system", "content": "Score the answer"},
                    {"role": "user", "content": "Answer: yes"},
                ],
            },
        )

        self.assertEqual(response.status_code, 200)
        completion = response.json()
        self.assertEqual(completion["choices"][0]["finish_reason"], "stop")
        self.assertEqual(json.loads(completion["choices"][0]["message"]["content"])["score"], 2)
        self.assertEqual(json.loads(completion["choices"][0]["message"]["content"])["correctness_probability"], 0.8)
        self.assertEqual(completion["usage"]["prompt_tokens"], 13)
        self.assertEqual(self.fake_agent.state, {"input": "user: Answer: yes"})
        self.assertEqual(self.fake_agent.questions["judge"]["instructions"], "Score the answer")
        self.assertEqual(self.fake_agent.questions["correctness"]["instructions"], laya_server.CORRECTNESS_INSTRUCTIONS)

    def test_correctness_guard_caps_scores_below_threshold(self):
        self.fake_agent.correctness_probability = 0.59
        response = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "laya",
                "messages": [
                    {"role": "system", "content": "Score the answer"},
                    {"role": "user", "content": "Answer: yes"},
                ],
            },
        )

        self.assertEqual(response.status_code, 200)
        content = json.loads(response.json()["choices"][0]["message"]["content"])
        self.assertEqual(content["raw_score"], 2)
        self.assertEqual(content["score"], 1)

    def test_correctness_guard_preserves_scores_at_threshold(self):
        self.fake_agent.correctness_probability = 0.6
        response = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "laya",
                "messages": [
                    {"role": "system", "content": "Score the answer"},
                    {"role": "user", "content": "Answer: yes"},
                ],
            },
        )

        self.assertEqual(response.status_code, 200)
        content = json.loads(response.json()["choices"][0]["message"]["content"])
        self.assertEqual(content["score"], 2)

    def test_streaming_completion_uses_sse_and_done_marker(self):
        response = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "laya",
                "messages": [
                    {"role": "system", "content": "Score the answer"},
                    {"role": "user", "content": "Answer: yes"},
                ],
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("text/event-stream", response.headers["content-type"])
        self.assertIn('"finish_reason": "stop"', response.text)
        self.assertIn('"prompt_tokens": 13', response.text)
        self.assertTrue(response.text.rstrip().endswith("data: [DONE]"))

    def test_chat_completion_accepts_advertised_tools_without_emitting_calls(self):
        response = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "laya",
                "messages": [
                    {"role": "system", "content": "Score the answer"},
                    {"role": "user", "content": "Answer: yes"},
                ],
                "tools": [{"type": "function", "function": {"name": "noop", "parameters": {"type": "object"}}}],
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["choices"][0]["finish_reason"], "stop")


class LayaServiceLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(laya_server.app)
        self.patches = [
            patch.object(laya_server, "agent", FakeAgent()),
            patch.object(laya_server, "IDLE_TIMEOUT_SECONDS", 60.0),
            patch.object(laya_server, "last_activity", time.monotonic() - 120),
        ]
        for active in self.patches:
            active.start()

    def tearDown(self):
        for active in self.patches:
            active.stop()

    def test_health_reports_starting_until_the_checkpoint_loads(self):
        with patch.object(laya_server, "agent", None), patch.object(laya_server, "device", None):
            self.assertEqual(self.client.get("/health").json()["status"], "starting")
        with patch.object(laya_server, "device", "mps"):
            self.assertEqual(
                self.client.get("/health").json(),
                {"status": "ok", "device": "mps", "checkpoint": laya_server.CHECKPOINT_ID},
            )

    def test_health_reports_the_configured_checkpoint(self):
        with patch.object(laya_server, "CHECKPOINT_ID", "/models/laya-ft"):
            self.assertEqual(self.client.get("/health").json()["checkpoint"], "/models/laya-ft")

    def test_api_requests_reset_the_idle_clock_but_health_checks_do_not(self):
        self.client.get("/health")
        self.assertTrue(laya_server.idle_expired(time.monotonic()), "health polling must not keep an idle service alive")
        self.assertEqual(self.client.get("/v1/models").status_code, 200)
        self.assertFalse(laya_server.idle_expired(time.monotonic()))

    def test_idle_shutdown_boundaries(self):
        start = laya_server.last_activity
        self.assertFalse(laya_server.idle_expired(start + 59.9))
        self.assertTrue(laya_server.idle_expired(start + 60))
        with laya_server.prediction_lock:
            self.assertFalse(laya_server.idle_expired(start + 600), "never exit mid-prediction")
        with patch.object(laya_server, "agent", None):
            self.assertFalse(laya_server.idle_expired(start + 600), "never exit while the checkpoint loads")
        with patch.object(laya_server, "IDLE_TIMEOUT_SECONDS", 0.0):
            self.assertFalse(laya_server.idle_expired(start + 600), "0 disables idle shutdown")


class LayaBatchPredictionTests(unittest.TestCase):
    QUESTIONS = {"destructive": {"type": "noul", "instructions": "Does the command delete data?"}}

    def setUp(self):
        self.client = TestClient(laya_server.app)
        self.agent_patch = patch.object(laya_server, "agent", FakeAgent())
        self.agent_patch.start()

    def tearDown(self):
        self.agent_patch.stop()

    def test_returns_one_result_per_state_in_request_order(self):
        response = self.client.post(
            "/v1/predict_batch",
            json={"states": ["ls -la", "rm -rf build", "cat notes.txt"], "questions": self.QUESTIONS},
        )

        self.assertEqual(response.status_code, 200)
        results = response.json()["results"]
        self.assertEqual([r["answers"]["destructive"]["noul"] for r in results], [0.1, 0.9, 0.1])
        self.assertEqual([r["usage"]["input_tokens"] for r in results], [6, 12, 13])

    def test_rejects_requests_while_the_checkpoint_loads(self):
        with patch.object(laya_server, "agent", None):
            response = self.client.post("/v1/predict_batch", json={"states": ["ls"], "questions": self.QUESTIONS})
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"], "Laya model is still starting")

    def test_rejects_an_empty_batch(self):
        response = self.client.post("/v1/predict_batch", json={"states": [], "questions": self.QUESTIONS})
        self.assertEqual(response.status_code, 400)

    def test_reports_model_failures_as_server_errors(self):
        with patch.object(FakeAgent, "predict_batch", side_effect=RuntimeError("boom")), \
                self.assertLogs("uvicorn.error.laya", level="ERROR"):
            response = self.client.post("/v1/predict_batch", json={"states": ["ls"], "questions": self.QUESTIONS})
        self.assertEqual(response.status_code, 500)

    def test_batch_requests_keep_the_service_alive(self):
        with patch.object(laya_server, "IDLE_TIMEOUT_SECONDS", 60.0), \
                patch.object(laya_server, "last_activity", time.monotonic() - 120):
            self.client.post("/v1/predict_batch", json={"states": ["ls"], "questions": self.QUESTIONS})
            self.assertFalse(laya_server.idle_expired(time.monotonic()))


if __name__ == "__main__":
    unittest.main()
