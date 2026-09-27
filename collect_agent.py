"""State-collection agent for RL: plays with a cheap behavior policy
(random, optionally mixed with the current model endpoint) and logs every
state it visits with per-action environment rewards.

Env vars:
  PACBRAIN_RL_LOG    JSONL output path (required to log)
  PACBRAIN_ENDPOINT  optional model endpoint; if set, used for ~half the moves
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pacai.core.agent

from rewards import action_rewards
from serializer import parse_action, serialize

class CollectAgent(pacai.core.agent.Agent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._log_path = os.environ.get("PACBRAIN_RL_LOG")
        self._client = None
        if os.environ.get("PACBRAIN_ENDPOINT"):
            import openai
            self._client = openai.OpenAI(
                base_url=os.environ["PACBRAIN_ENDPOINT"], api_key="none",
                timeout=20, max_retries=1)

    def get_action(self, state):
        prompt = serialize(state)
        rewards = action_rewards(state)
        if self._log_path:
            with open(self._log_path, "a") as f:
                f.write(json.dumps({"prompt": prompt, "rewards": rewards}) + "\n")

        legal = state.get_legal_actions()
        action = None
        if self._client is not None and self.rng.random() < 0.5:
            try:
                resp = self._client.completions.create(
                    model="pacman", prompt=prompt, max_tokens=6,
                    temperature=0.7, stop=["\n"])
                action = parse_action(resp.choices[0].text, legal)
            except Exception:
                action = None
        if action is None:
            action = self.rng.choice(list(legal))
        return action
