"""Scout agent: plays Pac-Man for the coach and logs every decision.

Env vars:
  COACH_STEP_LOG     JSONL path for per-decision rows (required)
  PACBRAIN_ENDPOINT  optional model endpoint; without it the scout plays randomly
  PACBRAIN_TRACE, PACBRAIN_BOARD, PACBRAIN_SEED
                     optional canonical trace for Memorable extraction (see contract.md)
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import openai
import pacai.core.agent

from memory.pacbrain import recorder_from_env
from rewards import bfs_dist
from serializer import parse_action, serialize


class ScoutAgent(pacai.core.agent.Agent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._log_path = os.environ["COACH_STEP_LOG"]
        endpoint = os.getenv("PACBRAIN_ENDPOINT")
        self._client = openai.OpenAI(base_url=endpoint, api_key=os.getenv("PACBRAIN_API_KEY") or "none",
                                     timeout=30, max_retries=1) if endpoint else None
        self._recorder = recorder_from_env(agent="coach-scout")
        self._turn = 0

    def get_action(self, state):
        legal = state.get_legal_actions()
        text, action, source = "", None, "random"
        if self._client is not None:
            try:
                response = self._client.completions.create(
                    model=os.getenv("PACBRAIN_SERVED_MODEL", "pacman"), prompt=serialize(state), max_tokens=6,
                    temperature=float(os.getenv("PACBRAIN_TEMP", "0.7")), stop=["\n"])
                text = response.choices[0].text
            except openai.OpenAIError as error:
                text = "<error: " + type(error).__name__ + ">"
            action = parse_action(text, legal)
            source = "policy" if action is not None else "fallback"
        if action is None:
            action = self.rng.choice(list(legal))
        pac = state.get_agent_position(0)
        row = {"t": self._turn, "score": state.score, "action": str(action), "source": source,
               "illegal_output": source == "fallback", "raw": text[:40],
               "legal": [str(a) for a in legal],
               "ghost_dist": bfs_dist(state.board, pac, list(state.get_nonscared_ghost_positions().values()))}
        with open(self._log_path, "a") as stream:
            stream.write(json.dumps(row) + "\n")
        if self._recorder:
            # Random scouting moves are not policy choices, so extraction stops at them.
            self._recorder.record(state, action, source, None)
        self._turn += 1
        return action

    def game_complete(self, final_state):
        if self._recorder:
            self._recorder.finish(final_state)
