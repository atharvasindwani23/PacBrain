"""CPU regressions for recorded results, prompts, and supervised target integrity."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

from gen_data import parse_summary
from make_results import summarize


class TrainingPipelineTests(unittest.TestCase):
    def test_engine_callback_failure_is_not_a_successful_game(self):
        with self.assertRaises(RuntimeError):
            parse_summary("Agent 0 crashed.\nScores: 973\nRecord: Win")
        with self.assertRaises(RuntimeError):
            parse_summary("partial game")
        self.assertEqual(parse_summary("Scores: -88\nRecord: Loss"), (-88, False))

    def test_summary_recomputes_and_rejects_inconsistent_aggregates(self):
        result = {"per_game": [
            {"seed": 1, "score": 10, "win": True, "moves": 2, "illegal": 1},
            {"seed": 2, "score": -4, "win": False, "moves": 4, "illegal": 0}]}
        measured = summarize(result)
        self.assertEqual(measured["avg_score"], 3)
        self.assertAlmostEqual(measured["illegal_move_rate"], 1 / 6)
        result["avg_score"] = 999
        with self.assertRaises(ValueError):
            summarize(result)

    def test_supervised_targets_are_never_silently_truncated(self):
        # Execute the pure helper without importing the optional Modal SDK.
        tree = ast.parse((Path(__file__).resolve().parents[1] / "modal_train.py").read_text())
        helper = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                      and node.name == "encode_pair")
        scope = {"MAX_LENGTH": 512}
        exec(compile(ast.Module(body=[helper], type_ignores=[]), "modal_train.py", "exec"), scope)
        class Tokenizer:
            eos_token_id = 99
            def __call__(self, text, **kwargs):
                return {"input_ids": list(range(len(text)))}
        encode = scope["encode_pair"]
        encoded = encode(Tokenizer(), {"prompt": "abc", "completion": "N"}, max_length=5)
        self.assertEqual(encoded["labels"], [-100, -100, -100, 0, 99])
        with self.assertRaises(ValueError):
            encode(Tokenizer(), {"prompt": "abc", "completion": "N"}, max_length=4)
        with self.assertRaises(ValueError):
            encode(Tokenizer(), {"prompt": "abc", "completion": ""})


try:
    import pacai.core.action
    from serializer import parse_action, serialize
    HAS_PACAI = True
except ImportError:
    HAS_PACAI = False


@unittest.skipUnless(HAS_PACAI, "Optional CPU PacAI dependency is not installed")
class SerializationTests(unittest.TestCase):
    def test_terminal_without_active_agent_and_wallless_dimensions(self):
        board = SimpleNamespace(height=2, width=3, get_walls=lambda: [],
                                get_marker_positions=lambda marker: [])
        def no_active_agent():
            raise ValueError("Cannot get legal actions when no agent is active")
        state = SimpleNamespace(board=board, game_over=True, get_food=lambda: [],
                                get_nonscared_ghost_positions=lambda: {},
                                get_scared_ghost_positions=lambda: {},
                                get_agent_position=lambda index: SimpleNamespace(row=1, col=2),
                                get_legal_actions=no_active_agent)
        prompt = serialize(state)
        self.assertIn("\n   \n  P\nLegal moves: \nMove:", prompt)

    def test_action_parser_rejects_reasoning_and_substrings(self):
        legal = [pacai.core.action.NORTH, pacai.core.action.EAST]
        self.assertEqual(parse_action(" North. ", legal), pacai.core.action.NORTH)
        for text in ("northeast", "do not go north", "north or east", "<think>east</think>", "south"):
            self.assertIsNone(parse_action(text, legal))


if __name__ == "__main__":
    unittest.main()
