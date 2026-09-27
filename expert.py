"""Scripted expert Pacman: BFS to nearest food, hard avoidance of non-scared
ghosts. With PACBRAIN_LOG set, records (prompt, completion) JSONL pairs for
fine-tuning. Referenced from the CLI as:  --pacman 'expert.py:ExpertAgent'
"""

import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pacai.core.action
import pacai.core.agent

from serializer import serialize

DANGER_RADIUS = 2


def bfs_dist(board, start, targets, max_depth=10000):
    """BFS distance from start to nearest position in targets, or None."""
    if not targets:
        return None, None
    targets = set(targets)
    seen = {start}
    queue = collections.deque([(start, 0)])
    while queue:
        pos, d = queue.popleft()
        if pos in targets:
            return d, pos
        if d >= max_depth:
            continue
        for _action, npos in board.get_neighbors(pos):
            if npos not in seen:
                seen.add(npos)
                queue.append((npos, d + 1))
    return None, None


class ExpertAgent(pacai.core.agent.Agent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._log_path = os.environ.get("PACBRAIN_LOG")

    def get_action(self, state):
        action = self._choose(state)
        if self._log_path:
            with open(self._log_path, "a") as f:
                f.write(json.dumps({
                    "prompt": serialize(state),
                    "completion": " " + str(action).capitalize(),
                }) + "\n")
        return action

    def _choose(self, state):
        board = state.board
        pac = state.get_agent_position(0)
        legal = list(state.get_legal_actions())
        if pacai.core.action.STOP in legal and len(legal) > 1:
            legal.remove(pacai.core.action.STOP)

        ghosts = list(state.get_nonscared_ghost_positions().values())
        food = set(state.get_food())
        scared = list(state.get_scared_ghost_positions().values())
        targets = food | set(scared)  # chasing scared ghosts is worth 200

        scored = []
        for action in legal:
            npos = pac.apply_action(action)
            gdist, _ = bfs_dist(board, npos, ghosts, max_depth=6)
            danger = gdist is not None and gdist <= DANGER_RADIUS
            tdist, _ = bfs_dist(board, npos, targets)
            tdist = tdist if tdist is not None else 9999
            # safe moves first, then nearest target, then farther from ghosts
            gd = gdist if gdist is not None else 99
            scored.append((danger, tdist, -gd, self.rng.random(), action))

        scored.sort()
        return scored[0][4]
