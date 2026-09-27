"""Pacman driven by an OpenAI-compatible completion endpoint, with optional memory.

PACBRAIN_MEMORY=none (default), local, or gbrain enables guarded recall.
PACBRAIN_TRACE records complete observations and executed actions for extraction.
PACBRAIN_API_KEY supplies an optional private endpoint credential (default: none).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import openai
import pacai.core.agent

from serializer import parse_action, serialize
from memory.pacbrain import memory_from_env, recorder_from_env
from memory.service import load_env


class LLMAgent(pacai.core.agent.Agent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        load_env()
        self._client = openai.OpenAI(
            base_url=os.environ["PACBRAIN_ENDPOINT"], api_key=os.getenv("PACBRAIN_API_KEY") or "none",
            timeout=30, max_retries=2)
        self._stats_path = os.environ.get("PACBRAIN_STATS")
        self._memory = memory_from_env()
        self._recorder = recorder_from_env(agent="llm")

    def get_action(self, state):
        legal = state.get_legal_actions()
        action = self._memory.choose(state) if self._memory else None
        decision = self._memory.decision() if self._memory else {
            "memory_provider": "none", "fallback_reason": "memory_disabled",
            "procedure_id": None, "procedure_step": None, "memory_error": None}
        text, illegal = "", False
        policy_called = action is None
        source = "procedure" if action is not None else "policy"
        if policy_called:
            try:
                resp = self._client.completions.create(
                    model="pacman", prompt=serialize(state), max_tokens=6,
                    temperature=float(os.environ.get("PACBRAIN_TEMP", "0")), stop=["\n"])
                text = resp.choices[0].text
                action = parse_action(text, legal)
            except Exception as error:
                # Avoid writing provider URLs, headers, or raw response bodies.
                text = "<error: " + type(error).__name__ + ">"
            illegal = action is None
            if illegal:
                action = self.rng.choice([a for a in legal])
                source = "fallback"
        decision.update(policy_called=policy_called)
        if source == "procedure":
            decision["fallback_reason"] = None
        if self._recorder:
            self._recorder.record(state, action, source, decision)
        if self._stats_path:
            with open(self._stats_path, "a") as stream:
                stream.write(json.dumps(dict(decision, raw=text[:80], illegal=illegal,
                                             action=str(action), source=source,
                                             score=state.score)) + "\n")
        return action

    def game_complete(self, final_state):
        if self._recorder:
            self._recorder.finish(final_state)
