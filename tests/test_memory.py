import copy
import tempfile
import unittest

from memory.core import MemoryStore, ProcedureCursor, extract_procedure, nearest_danger


def trace():
    return {"episode_id": "fixture-1", "layout": "testCorridor", "seed": 1,
            "observation_timing": "pre_action", "frames": [
                {"grid": "%%%%%%%%%\n%P.... G%\n%%%%%%%%%", "action": "E", "source": "policy", "score": 0},
                {"grid": "%%%%%%%%%\n% P...G %\n%%%%%%%%%", "action": "E", "source": "policy", "score": 9},
                {"grid": "%%%%%%%%%\n%  P.G  %\n%%%%%%%%%", "action": "W", "source": "policy", "score": 18}],
            "result": {"score": 18, "died": False, "cleared": False}}


class MemoryTests(unittest.TestCase):
    def test_extract_and_replay_observed_moves(self):
        source = trace()
        procedure = extract_procedure(source)
        self.assertEqual([s["action"] for s in procedure["steps"]], ["E", "E"])
        cursor = ProcedureCursor(procedure)
        self.assertEqual(cursor.next_action(source["frames"][0], "testCorridor"), "E")
        self.assertEqual(cursor.next_action(source["frames"][1], "testCorridor"), "E")
        self.assertIsNone(cursor.next_action(source["frames"][2], "testCorridor"))
        self.assertEqual(cursor.reason, "complete")

    def test_stochastic_ghost_aborts_without_consuming_more_steps(self):
        source = trace()
        cursor = ProcedureCursor(extract_procedure(source))
        observation = copy.deepcopy(source["frames"][0])
        observation["grid"] = "%%%%%%%%%\n%P.G... %\n%%%%%%%%%"
        self.assertIsNone(cursor.next_action(observation))
        self.assertEqual(cursor.reason, "ghost_nearby")
        self.assertIsNone(cursor.next_action(source["frames"][0]))

    def test_scared_ghost_is_not_danger(self):
        self.assertIsNone(nearest_danger({"grid": "%%%%%\n%Pg %\n%%%%%"}))

    def test_layout_and_position_guards(self):
        source = trace()
        cursor = ProcedureCursor(extract_procedure(source))
        self.assertIsNone(cursor.next_action(source["frames"][1]))
        self.assertEqual(cursor.reason, "position_diverged")
        cursor = ProcedureCursor(extract_procedure(source))
        self.assertIsNone(cursor.next_action(source["frames"][0], "anotherLayout"))
        self.assertEqual(cursor.reason, "layout_changed")

    def test_does_not_learn_random_or_unobserved_final_move(self):
        source = trace()
        source["frames"][0]["source"] = "random"
        with self.assertRaises(ValueError):
            extract_procedure(source)
        source = trace()
        source["frames"] = source["frames"][:1]
        with self.assertRaises(ValueError):
            extract_procedure(source)

    def test_roundtrip_and_provider_id_filter(self):
        with tempfile.TemporaryDirectory() as folder:
            store = MemoryStore(folder)
            procedure = extract_procedure(trace())
            store.save(procedure)
            self.assertEqual(store.recall("testCorridor")["id"], procedure["id"])
            self.assertIsNone(store.recall("testCorridor", allowed_ids=[]))

    def test_reject_post_action_frames(self):
        source = trace()
        source["observation_timing"] = "post_action"
        with self.assertRaises(ValueError):
            extract_procedure(source)


if __name__ == "__main__":
    unittest.main()
