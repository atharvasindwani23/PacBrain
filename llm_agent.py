"""Pacman agent driven by an LLM behind an OpenAI-compatible /v1/completions
endpoint. Env vars:
  PACBRAIN_ENDPOINT  base URL, e.g. https://...modal.run/v1
  PACBRAIN_STATS     optional path; appends per-move JSONL stats
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import openai
import pacai.core.agent

from serializer import parse_action, serialize


class LLMAgent(pacai.core.agent.Agent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._client = openai.OpenAI(
            base_url=os.environ["PACBRAIN_ENDPOINT"], api_key="none",
            timeout=30, max_retries=2)
        self._stats_path = os.environ.get("PACBRAIN_STATS")

    def get_action(self, state):
        prompt = serialize(state)
        legal = state.get_legal_actions()
        text, action = "", None
        try:
            resp = self._client.completions.create(
                model="pacman", prompt=prompt, max_tokens=6,
                temperature=float(os.environ.get("PACBRAIN_TEMP", "0")), stop=["\n"])
            text = resp.choices[0].text
            action = parse_action(text, legal)
        except Exception as e:
            text = f"<error: {e}>"
        illegal = action is None
        if illegal:
            action = self.rng.choice([a for a in legal])
        if self._stats_path:
            with open(self._stats_path, "a") as f:
                f.write(json.dumps({"raw": text[:80], "illegal": illegal,
                                    "action": str(action)}) + "\n")
        return action
