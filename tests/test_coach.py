"""Offline tests for the coach: no model endpoints, Memorable or GBrain needed."""

import dataclasses
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from coach.designer import design, write_design
from coach.evals import compare, load_metrics, validate_eval_suite
from coach.games import flappy, load_game
from coach.memory_io import load_local_notes, note_page, parse_page
from coach.rewards import compile_reward, load_spec, validate_reward_spec
from coach.scout import digest, scout, validate_notes

HAS_PACAI = importlib.util.find_spec("pacai") is not None
ROOT = Path(__file__).resolve().parents[1]

FLAPPY_REWARD = {
    "terms": [
        {"name": "crash", "weight": -50, "why": "Every scouted game crashed (7000-0, 7001-0)."},
        {"name": "pipe_passed", "weight": 20, "why": "Pipes are the score (7000-1)."},
        {"name": "gap_alignment", "weight": 10, "why": "Crashes happen off-centre (7001-0)."},
    ],
    "scale": 10, "clip": 3, "illegal_output": -2,
}
FLAPPY_EVALS = [
    {"metric": "avg_pipes", "target": 3, "why": "7000-1"},
    {"metric": "crash_rate", "target": 0.5, "why": "7000-0"},
]
SCOUT_REPLY = {"notes": [
    {"kind": "failure", "text": "The game ended in a crash.", "evidence": ["died", "crash_type"]},
    {"kind": "pattern", "text": "The bird rarely flapped.", "evidence": ["flap_rate"]},
]}


class FakeLLM:
    model = "fake-coach"

    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests = []

    def complete(self, system, user):
        self.requests.append(json.loads(user))
        return self.replies.pop(0)


class FlappyGameTest(unittest.TestCase):
    def test_reset_is_deterministic(self):
        self.assertEqual(flappy.reset(11), flappy.reset(11))
        self.assertNotEqual(flappy.reset(11).pipes, flappy.reset(12).pipes)

    def test_never_flapping_falls_to_the_floor(self):
        state = flappy.reset(1)
        while state.alive:
            state = flappy.step(state, "WAIT")
        self.assertGreaterEqual(state.y, flappy.HEIGHT)
        with self.assertRaises(ValueError):
            flappy.step(state, "WAIT")

    def test_random_scout_is_reproducible(self):
        first = flappy.play_episodes([3, 4])
        second = flappy.play_episodes([3, 4])
        self.assertEqual([e["result"] for e in first], [e["result"] for e in second])
        self.assertTrue(all(step["source"] == "random" for e in first for step in e["steps"]))

    def test_policy_output_is_parsed_strictly(self):
        episodes = flappy.play_episodes([5], policy=lambda prompt: "I think flap", max_turns=20)
        self.assertTrue(all(step["illegal_output"] for step in episodes[0]["steps"]))
        self.assertEqual(flappy.parse_action(" Flap. "), "FLAP")
        self.assertIsNone(flappy.parse_action("flapwait"))

    def test_observation_is_a_move_prompt(self):
        prompt = flappy.observe(flappy.reset(2))
        self.assertIn("B", prompt)
        self.assertTrue(prompt.endswith("Legal moves: flap, wait\nMove:"))


class RewardSpecTest(unittest.TestCase):
    def test_compiles_to_the_trainer_reward_table(self):
        table = compile_reward(FLAPPY_REWARD, flappy)(flappy.reset(3))
        self.assertEqual(set(table), {"FLAP", "WAIT"})
        self.assertTrue(all(-3 <= value <= 3 for value in table.values()))

    def test_crashing_scores_below_surviving(self):
        state = dataclasses.replace(flappy.reset(3), y=97.0, vy=5.0)
        table = compile_reward(FLAPPY_REWARD, flappy)(state)
        self.assertLess(table["WAIT"], table["FLAP"])

    def test_rejects_unsafe_specs(self):
        bad = [
            dict(FLAPPY_REWARD, terms=[{"name": "teleport", "weight": 1}]),
            dict(FLAPPY_REWARD, terms=[{"name": "crash", "weight": float("inf")}]),
            dict(FLAPPY_REWARD, terms=FLAPPY_REWARD["terms"] + [FLAPPY_REWARD["terms"][0]]),
            dict(FLAPPY_REWARD, illegal_output=0),
            dict(FLAPPY_REWARD, clip=True),
        ]
        for spec in bad:
            with self.assertRaises(ValueError):
                validate_reward_spec(spec, flappy)

    @unittest.skipUnless(HAS_PACAI, "edq-pacai is not installed")
    def test_handwritten_pacman_reward_is_a_valid_spec(self):
        pacman = load_game("pacman")
        spec = load_spec(ROOT / "coach/specs/pacman_handwritten.json", "pacman")
        self.assertEqual(validate_reward_spec(spec, pacman)["illegal_output"], -1.5)


class ScoutTest(unittest.TestCase):
    def test_notes_must_cite_the_digest(self):
        episode = flappy.play_episodes([7000])[0]
        facts = digest(flappy, episode)
        notes = validate_notes(SCOUT_REPLY, facts)
        self.assertEqual(notes[0]["values"]["crash_type"], facts["crash_type"])
        with self.assertRaises(ValueError):
            validate_notes({"notes": [{"kind": "failure", "text": "Bad luck.", "evidence": ["vibes"]}]}, facts)
        with self.assertRaises(ValueError):
            validate_notes({"notes": []}, facts)

    def test_scout_writes_notes_that_read_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            llm = FakeLLM(SCOUT_REPLY, SCOUT_REPLY)
            result = scout("flappy", 2, 7000, llm, out_dir=tmp, memory="none")
            self.assertEqual(result["episodes"], 2)
            self.assertIsNone(result["gbrain"])
            pages = load_local_notes(Path(tmp) / "memory", "flappy")
            self.assertEqual([p["seed"] for p in pages], [7000, 7001])
            self.assertEqual(llm.requests[0]["digest"]["seed"], 7000)

    def test_note_page_round_trips(self):
        episode = flappy.play_episodes([9])[0]
        facts = digest(flappy, episode)
        notes = validate_notes(SCOUT_REPLY, facts)
        payload = parse_page(note_page("flappy", episode, facts, notes))
        self.assertEqual(payload["notes"], notes)
        self.assertEqual(payload["kind"], "scout-notes")


class DesignerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        scout("flappy", 2, 7000, FakeLLM(SCOUT_REPLY, SCOUT_REPLY), out_dir=self.tmp.name, memory="none")
        self.pages = load_local_notes(Path(self.tmp.name) / "memory", "flappy")

    def tearDown(self):
        self.tmp.cleanup()

    def test_designs_a_grounded_reward_and_eval_suite(self):
        llm = FakeLLM({"reward": FLAPPY_REWARD, "evals": FLAPPY_EVALS, "rationale": "Stop crashing."})
        result = design("flappy", self.pages, llm)
        self.assertEqual(result["cited_notes"], ["7000-0", "7000-1", "7001-0"])
        self.assertEqual(llm.requests[0]["stats"]["episodes"], 2)
        paths = write_design(result, Path(self.tmp.name) / "generated", Path(self.tmp.name) / "memory")
        spec = load_spec(paths["reward_spec"], "flappy")
        self.assertEqual(validate_reward_spec(spec, flappy)["terms"][0]["name"], "crash")
        self.assertTrue((Path(self.tmp.name) / "memory/designs/flappy/latest.md").is_file())

    def test_rejects_an_ungrounded_design(self):
        uncited = dict(FLAPPY_REWARD, terms=[{"name": "alive", "weight": 1, "why": "seems good"}])
        llm = FakeLLM({"reward": uncited, "evals": [{"metric": "avg_pipes", "target": 1, "why": ""},
                                                    {"metric": "crash_rate", "target": 1, "why": ""}]})
        with self.assertRaises(ValueError):
            design("flappy", self.pages, llm)

    def test_rejects_unknown_eval_metrics(self):
        with self.assertRaises(ValueError):
            validate_eval_suite([{"metric": "vibes", "target": 1}, {"metric": "avg_pipes", "target": 1}], flappy)


class TrainingDataTest(unittest.TestCase):
    def test_flappy_states_match_the_grpo_format(self):
        rows = flappy.training_states(range(5), compile_reward(FLAPPY_REWARD, flappy), max_states=40)
        self.assertEqual(len(rows), 40)
        self.assertEqual(len({row["prompt"] for row in rows}), 40)
        self.assertTrue(all(set(row["rewards"]) == {"FLAP", "WAIT"} for row in rows))


@unittest.skipUnless(HAS_PACAI, "edq-pacai is not installed")
class PacmanReportTest(unittest.TestCase):
    def test_compares_recorded_runs_against_a_suite(self):
        pacman = load_game("pacman")
        suite = validate_eval_suite([
            {"metric": "illegal_move_rate", "target": 0.05},
            {"metric": "pellets_per_game", "target": 11},
            {"metric": "death_rate", "target": 0.5},
        ], pacman)
        rows = {row["metric"]: row for row in compare(suite, load_metrics(ROOT / "results/base.json", pacman),
                                                      load_metrics(ROOT / "results/rl.json", pacman))}
        self.assertTrue(rows["illegal_move_rate"]["improved"] and rows["illegal_move_rate"]["tuned_passed"])
        self.assertAlmostEqual(rows["pellets_per_game"]["tuned"], 11.6)
        self.assertFalse(rows["death_rate"]["improved"])


if __name__ == "__main__":
    unittest.main()
