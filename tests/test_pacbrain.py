import contextlib
import importlib.util
import io
import json
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from memory.core import extract_procedure
from memory.pacbrain import EpisodeRecorder, MemoryPolicy, audit_legacy_trace, parse_prompt
from tests.test_memory import trace


def prompt(grid):
    return "You are Pacman. Grid: % wall.\n" + grid + "\nLegal moves: STOP, EAST, WEST\nMove:"


class State:
    def __init__(self, grid, score=0, history=(), terminal=False, alive=True):
        self.prompt = prompt(grid)
        self.score, self.game_over, self.alive = score, terminal, alive
        self.history = list(history)
        self.turn_count = len(history) * 3

    def food_count(self):
        return self.prompt.count(".")

    def get_agent_position(self, index):
        return (1, 1) if self.alive else None

    def get_agent_actions(self, index):
        return self.history

    def get_legal_actions(self):
        return ["EAST", "WEST", "STOP"]


class PacbrainTests(unittest.TestCase):
    def serializer(self):
        return patch.dict("sys.modules", {"serializer": types.SimpleNamespace(serialize=lambda state: state.prompt)})

    def test_prompt_recovers_exact_grid_and_cardinal_legality(self):
        grid = "%%%%%\n% P.%\n%%%%%"
        self.assertEqual(parse_prompt(prompt(grid)), {"grid": grid, "legal_moves": ["E", "W"]})
        with self.assertRaisesRegex(ValueError, "rectangular"):
            parse_prompt(prompt("%%%\n%P.%"))

    def test_legacy_trace_refuses_invented_rewards(self):
        report = audit_legacy_trace({"board": "classic-small", "seed": 1, "score": 999,
                                    "steps": [{"prompt": prompt("%%%%%\n%P..%\n%%%%%"), "completion": " East"}]})
        self.assertEqual(report["missing_step_scores"], 1)
        self.assertFalse(report["replayable"])

    def test_recorder_produces_extractable_real_trace_and_confirms_actions(self):
        frames = trace()["frames"]
        states = [State(frame["grid"], frame["score"], ["EAST"] * index, terminal=index == 2)
                  for index, frame in enumerate(frames)]
        with tempfile.TemporaryDirectory() as folder, self.serializer():
            recorder = EpisodeRecorder(Path(folder) / "episode.json", "testCorridor", seed=4)
            recorder.record(states[0], "EAST", "policy", {"policy_called": True})
            recorder.record(states[1], "EAST", "procedure", {"policy_called": False})
            recorded = recorder.finish(states[2])
            self.assertEqual(json.loads(recorder.path.read_text()), recorded)
        self.assertEqual([frame["action_ok"] for frame in recorded["frames"]], [True, True])
        self.assertFalse(recorded["synthetic"])
        self.assertEqual(recorded["result"]["llm_calls"], 1)
        self.assertEqual(recorded["result"]["procedure_steps"], 1)
        self.assertEqual(recorded["result"]["turns"], 2)
        self.assertEqual(len(extract_procedure(recorded)["steps"]), 2)

    def test_recorder_uses_actual_engine_action_after_correction(self):
        first = State(trace()["frames"][0]["grid"])
        final = State(trace()["frames"][0]["grid"], -1, ["WEST"], terminal=True)
        with tempfile.TemporaryDirectory() as folder, self.serializer():
            recorder = EpisodeRecorder(Path(folder) / "episode.json", "testCorridor")
            recorder.record(first, "EAST")
            recorded = recorder.finish(final)
        frame = recorded["frames"][0]
        self.assertEqual((frame["action"], frame["source"], frame["requested_action"]), ("W", "fallback", "E"))
        self.assertFalse(frame["action_ok"])

    def test_death_trace_keeps_terminal_board_without_pacman(self):
        initial = State(trace()["frames"][0]["grid"])
        terminal = State("%%%%%%%%%\n% G.... %\n%%%%%%%%%", -501, ["EAST"], terminal=True, alive=False)
        with tempfile.TemporaryDirectory() as folder, self.serializer():
            recorder = EpisodeRecorder(Path(folder) / "episode.json", "testCorridor")
            recorder.record(initial, "EAST")
            recorded = recorder.finish(terminal)
        self.assertTrue(recorded["terminal_observation"]["died"])
        self.assertTrue(recorded["result"]["died"])
        with self.assertRaisesRegex(ValueError, "no positively rewarded"):
            extract_procedure(recorded)

    def test_evaluator_counts_model_calls_and_confirmed_memory_actions(self):
        import importlib
        evaluator = importlib.import_module("eval")
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "results").mkdir()
            def run(command, **kwargs):
                env = kwargs["env"]
                self.assertEqual(env["PACBRAIN_MEMORY"], "local")
                rows = [{"illegal": False, "policy_called": False, "source": "procedure"},
                        {"illegal": False, "policy_called": True, "source": "policy"},
                        {"illegal": True, "policy_called": True, "source": "fallback"}]
                Path(env["PACBRAIN_STATS"]).write_text("".join(json.dumps(row) + "\n" for row in rows))
                Path(env["PACBRAIN_TRACE"]).write_text(json.dumps({"result": {"procedure_steps": 1}}))
                return types.SimpleNamespace(stdout="Scores: 18\nRecord: Win", stderr="", returncode=0)
            with patch.object(evaluator, "ROOT", folder), patch.object(evaluator.subprocess, "run", side_effect=run):
                result = evaluator.run_game((2000, "https://example.invalid/v1", "trial", False,
                                             "local", folder, "classic-small", folder))
        self.assertEqual((result["moves"], result["policy_calls"], result["procedure_steps"], result["fallback_steps"]),
                         (3, 2, 1, 1))
        self.assertEqual(result["illegal"], 1)

    def test_evaluator_blocks_existing_summary_gif_and_trace_without_overwrite(self):
        import importlib
        evaluator = importlib.import_module("eval")
        for relative in ("results/base.json", "results/base_seed2000.gif",
                         "results/base_traces/base_seed2000.json"):
            with self.subTest(artifact=relative), tempfile.TemporaryDirectory() as folder:
                target = Path(folder, relative)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("existing evidence")
                with patch.object(evaluator, "ROOT", folder), patch.object(evaluator, "run_game") as run, \
                        contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                    evaluator.main(["--endpoint", "https://example.invalid/v1", "--label", "base",
                                    "--games", "1", "--gifs", "0"])
                self.assertEqual(raised.exception.code, 2)
                self.assertEqual(target.read_text(), "existing evidence")
                run.assert_not_called()

    def test_explicit_overwrite_clears_stale_gif_and_all_failed_average_is_null(self):
        import importlib
        evaluator = importlib.import_module("eval")
        failed = {"seed": 2000, "score": 18, "win": False, "moves": 1, "illegal": 1,
                  "policy_calls": 1, "procedure_steps": 0, "fallback_steps": 1,
                  "memory_failures": 0, "error": "missing_trace", "trace": None}
        with tempfile.TemporaryDirectory() as folder:
            results = Path(folder, "results")
            results.mkdir()
            (results / "base.json").write_text("old summary")
            stale = results / "base_seed2000.gif"
            stale.write_bytes(b"old animation")
            with patch.object(evaluator, "ROOT", folder), patch.object(evaluator, "run_game", return_value=failed), \
                    contextlib.redirect_stdout(io.StringIO()):
                status = evaluator.main(["--endpoint", "https://example.invalid/v1", "--label", "base",
                                         "--games", "1", "--gifs", "0", "--overwrite"])
            summary = json.loads((results / "base.json").read_text())
            self.assertFalse(stale.exists())
        self.assertEqual(status, 1)
        self.assertIsNone(summary["avg_score"])
        self.assertIsNone(summary["win_rate"])
        self.assertEqual(summary["completed_games"], 0)

    def test_evaluator_reports_requested_missing_gif_as_failure(self):
        import importlib
        evaluator = importlib.import_module("eval")
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "results").mkdir()
            def run(command, **kwargs):
                env = kwargs["env"]
                Path(env["PACBRAIN_TRACE"]).write_text(json.dumps({"result": {"procedure_steps": 0}}))
                return types.SimpleNamespace(stdout="Scores: 0\nRecord: Loss", stderr="", returncode=0)
            with patch.object(evaluator, "ROOT", folder), patch.object(evaluator.subprocess, "run", side_effect=run):
                result = evaluator.run_game((2000, "https://example.invalid/v1", "trial", True,
                                             "none", folder, "classic-small", folder))
        self.assertEqual(result["error"], "missing_gif")
        self.assertIsNone(result["gif"])

    def test_evaluator_rejects_agent_crash_even_with_successful_process_and_score(self):
        import importlib
        evaluator = importlib.import_module("eval")
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "results").mkdir()
            def run(command, **kwargs):
                Path(kwargs["env"]["PACBRAIN_TRACE"]).write_text(json.dumps({"result": {"procedure_steps": 0}}))
                return types.SimpleNamespace(stdout="Scores: -500\nRecord: Loss", stderr="Agent 0 crashed", returncode=0)
            with patch.object(evaluator, "ROOT", folder), patch.object(evaluator.subprocess, "run", side_effect=run):
                result = evaluator.run_game((2000, "https://example.invalid/v1", "trial", False,
                                             "none", folder, "classic-small", folder))
        self.assertEqual(result["error"], "agent_crashed")

    def test_guarded_memory_uses_one_recall_then_aborts_for_new_ghost(self):
        source = trace()
        procedure = extract_procedure(source)
        state = State(source["frames"][0]["grid"])
        danger = State("%%%%%%%%%\n% PG... %\n%%%%%%%%%", score=9, history=["EAST"])
        with self.serializer(), patch("memory.pacbrain.recall_procedure", return_value={"procedure": procedure}) as recall:
            policy = MemoryPolicy("testCorridor")
            self.assertEqual(policy.choose(state), "EAST")
            self.assertIsNone(policy.choose(danger))
            self.assertEqual(policy.reason, "ghost_nearby")
            self.assertIsNone(policy.choose(State(source["frames"][1]["grid"])))
            recall.assert_called_once()

    def test_terminal_state_never_queries_memory(self):
        with self.serializer(), patch("memory.pacbrain.recall_procedure") as recall:
            policy = MemoryPolicy("testCorridor")
            self.assertIsNone(policy.choose(State(trace()["frames"][0]["grid"], terminal=True)))
            self.assertEqual(policy.reason, "episode_finished")
            recall.assert_not_called()


@unittest.skipUnless(importlib.util.find_spec("pacai") and importlib.util.find_spec("openai"),
                     "gameplay integration requires requirements.txt")
class LLMAgentTests(unittest.TestCase):
    def test_optional_private_endpoint_credential_and_public_default(self):
        import llm_agent
        for key, expected in ((None, "none"), ("test-private-key", "test-private-key")):
            environment = {"PACBRAIN_ENDPOINT": "https://example.invalid/v1", "PACBRAIN_MEMORY": "none"}
            if key:
                environment["PACBRAIN_API_KEY"] = key
            with self.subTest(private=key is not None), patch.dict(os.environ, environment, clear=True), \
                    patch.object(llm_agent, "load_env"), patch.object(llm_agent.openai, "OpenAI") as client, \
                    patch.object(llm_agent, "recorder_from_env", return_value=None):
                llm_agent.LLMAgent()
                self.assertEqual(client.call_args.kwargs["api_key"], expected)

    def test_disabled_memory_keeps_model_path_and_replay_skips_one_call(self):
        import llm_agent
        import pacai.core.action

        state = State(trace()["frames"][0]["grid"])
        state.get_legal_actions = lambda: [pacai.core.action.EAST, pacai.core.action.WEST]
        client = types.SimpleNamespace(completions=types.SimpleNamespace())
        response = types.SimpleNamespace(choices=[types.SimpleNamespace(text=" East")])
        from unittest.mock import Mock
        client.completions.create = Mock(return_value=response)
        environment = {"PACBRAIN_ENDPOINT": "https://example.invalid/v1", "PACBRAIN_MEMORY": "none"}
        with patch.dict(os.environ, environment), patch.object(llm_agent.openai, "OpenAI", return_value=client), \
                patch.object(llm_agent, "recorder_from_env", return_value=None), \
                patch.object(llm_agent, "serialize", return_value=state.prompt):
            agent = llm_agent.LLMAgent()
            self.assertEqual(agent.get_action(state), pacai.core.action.EAST)
            client.completions.create.assert_called_once()
            agent._memory = types.SimpleNamespace(choose=lambda current: pacai.core.action.WEST,
                                                   decision=lambda: {"fallback_reason": "recalled_step"})
            self.assertEqual(agent.get_action(state), pacai.core.action.WEST)
            client.completions.create.assert_called_once()


if __name__ == "__main__":
    unittest.main()
