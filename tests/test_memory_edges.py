"""Boundary regressions for malformed storage and terminal game observations."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from memory.core import MemoryStore, ProcedureCursor, extract_procedure, validate_procedure


def corridor_trace():
    return {"episode_id": "edge-episode", "layout": "testCorridor", "seed": 1,
            "observation_timing": "pre_action", "frames": [
                {"grid": "%%%%%%%%%\n%P.... G%\n%%%%%%%%%", "action": "E", "source": "policy", "score": 0},
                {"grid": "%%%%%%%%%\n% P...G %\n%%%%%%%%%", "action": "E", "source": "policy", "score": 9},
                {"grid": "%%%%%%%%%\n%  P.G  %\n%%%%%%%%%", "action": "E", "source": "policy", "score": 18}],
            "result": {"score": 18, "died": False, "cleared": False}}


def empty_corridor_trace(scores):
    width = len(scores) + 4
    frames = []
    for index, score in enumerate(scores):
        row = ["%"] + [" "] * (width - 2) + ["%"]
        row[index + 1] = "P"
        frames.append({"grid": ["%" * width, "".join(row), "%" * width],
                       "action": "E", "source": "policy", "score": score})
    return {"episode_id": "empty-cells", "layout": "emptyCorridor", "frames": frames,
            "result": {"score": scores[-1], "died": False, "cleared": False}}


class MemoryEdgeTests(unittest.TestCase):
    def test_terminal_without_p_preserves_verified_prefix(self):
        source = corridor_trace()
        source["result"] = {"score": -482, "died": True, "cleared": False}
        # A renderer may draw the ghost over Pac-Man after a terminal collision.
        source["terminal_observation"] = {"grid": "%%%%%%%%%\n%   G   %\n%%%%%%%%%", "score": -482}
        procedure = extract_procedure(source)
        self.assertEqual([step["action"] for step in procedure["steps"]], ["E", "E"])

    def test_invalid_terminal_without_death_flag_preserves_verified_prefix(self):
        source = corridor_trace()
        source["terminal_observation"] = {"grid": "%%%%%%%%%\n%   G   %\n%%%%%%%%%", "score": -482}
        self.assertEqual(len(extract_procedure(source)["steps"]), 2)

    def test_malformed_json_procedure_is_skipped_during_recall(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(directory)
            valid = extract_procedure(corridor_trace())
            store.save(valid)
            # Valid JSON, but not a valid procedure; must not take down all recall.
            Path(directory, "broken.json").write_text(json.dumps({"version": 1, "steps": ["bad"]}))
            self.assertEqual(store.recall("testCorridor")["id"], valid["id"])

    def test_non_object_observation_aborts_cursor(self):
        cursor = ProcedureCursor(extract_procedure(corridor_trace()))
        self.assertIsNone(cursor.next_action(None))
        self.assertTrue(cursor.stopped)
        self.assertEqual(cursor.reason, "invalid_observation_or_procedure")

    def test_negative_stored_danger_distance_cannot_disable_ghost_guard(self):
        source = corridor_trace()
        malformed = extract_procedure(source)
        malformed["danger_distance"] = -1
        observation = copy.deepcopy(source["frames"][0])
        observation["grid"] = "%%%%%%%%%\n%PG.... %\n%%%%%%%%%"
        cursor = ProcedureCursor(malformed)
        self.assertIsNone(cursor.next_action(observation))
        self.assertTrue(cursor.stopped)

    def test_empty_cell_cost_between_rewarded_steps_is_retained(self):
        procedure = extract_procedure(empty_corridor_trace([0, 9, 8, 17]))
        self.assertEqual([step["score_delta"] for step in procedure["steps"]], [9, -1, 9])
        self.assertEqual(procedure["observed_reward"], 17)

    def test_initial_empty_cell_cost_can_lead_to_reward(self):
        procedure = extract_procedure(empty_corridor_trace([0, -1, 8]))
        self.assertEqual([step["score_delta"] for step in procedure["steps"]], [-1, 9])

    def test_unrewarded_suffix_is_trimmed_to_a_positive_prefix(self):
        procedure = extract_procedure(empty_corridor_trace([0, 2, 1, 0]))
        self.assertEqual([step["score_delta"] for step in procedure["steps"]], [2, -1])
        self.assertEqual(procedure["observed_reward"], 1)

    def test_large_negative_reward_stops_before_failed_transition(self):
        procedure = extract_procedure(empty_corridor_trace([0, 9, -491]))
        self.assertEqual(len(procedure["steps"]), 1)
        self.assertEqual(procedure["observed_reward"], 9)

    def test_explicit_failed_action_stops_before_transition(self):
        source = empty_corridor_trace([0, 9, 18])
        source["frames"][1]["action_ok"] = False
        self.assertEqual(len(extract_procedure(source)["steps"]), 1)

    def test_nonfinite_or_too_small_danger_distances_are_rejected(self):
        for distance in (-1, 0, 1, True, float("nan"), float("inf")):
            with self.subTest(distance=distance), self.assertRaises(ValueError):
                extract_procedure(corridor_trace(), danger_distance=distance)

    def test_malformed_reward_is_skipped_before_ranking(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryStore(directory)
            valid = extract_procedure(corridor_trace())
            store.save(valid)
            for index, reward in enumerate(("nine", float("nan"), float("inf"))):
                broken = copy.deepcopy(valid)
                broken["observed_reward"] = reward
                Path(directory, "broken-%d.json" % index).write_text(json.dumps(broken))
            self.assertEqual(store.procedures(), [valid])
            self.assertEqual(store.recall("testCorridor")["id"], valid["id"])

    def test_stored_transition_must_match_action(self):
        malformed = extract_procedure(corridor_trace())
        malformed["steps"][0]["next_position"] = [7, 1]
        with self.assertRaises(ValueError):
            validate_procedure(malformed)


if __name__ == "__main__":
    unittest.main()
