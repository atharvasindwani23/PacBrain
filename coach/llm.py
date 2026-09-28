"""Model clients for the coach.

CoachLLM is the coach's own reasoning model (episode notes, reward design): any
OpenAI-compatible chat endpoint, configured with COACH_ENDPOINT, COACH_MODEL and
optionally COACH_API_KEY. CompletionPolicy is the model being trained, served the
same way llm_agent.py calls it.
"""

import json
import os

import openai


class CoachLLM:
    def __init__(self, endpoint=None, model=None, api_key=None, timeout=90):
        endpoint = endpoint or os.getenv("COACH_ENDPOINT")
        model = model or os.getenv("COACH_MODEL")
        if not endpoint or not model:
            raise RuntimeError("Set COACH_ENDPOINT and COACH_MODEL (any OpenAI-compatible chat "
                               "endpoint) so the coach can write notes and design rewards.")
        self.model = model
        self._client = openai.OpenAI(base_url=endpoint, api_key=api_key or os.getenv("COACH_API_KEY") or "none",
                                     timeout=timeout, max_retries=2)

    def complete(self, system, user):
        """One JSON-object reply. Invalid JSON raises instead of being guessed at."""
        response = self._client.chat.completions.create(
            model=self.model, temperature=0.2, response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        text = response.choices[0].message.content or ""
        try:
            return json.loads(text)
        except json.JSONDecodeError as error:
            raise ValueError(f"Coach model {self.model} returned invalid JSON: {error}") from error


class CompletionPolicy:
    """The policy under training: prompt in, raw move text out."""

    def __init__(self, endpoint, temperature=None, served_model=None):
        self._client = openai.OpenAI(base_url=endpoint, api_key=os.getenv("PACBRAIN_API_KEY") or "none",
                                     timeout=30, max_retries=1)
        self._temperature = float(temperature if temperature is not None else os.getenv("PACBRAIN_TEMP", "0.7"))
        self._model = served_model or os.getenv("PACBRAIN_SERVED_MODEL", "pacman")

    def __call__(self, prompt):
        try:
            response = self._client.completions.create(model=self._model, prompt=prompt, max_tokens=6,
                                                       temperature=self._temperature, stop=["\n"])
        except openai.OpenAIError as error:
            # Counted as an illegal output by the caller; never logs URLs or bodies.
            return "<error: " + type(error).__name__ + ">"
        return response.choices[0].text
