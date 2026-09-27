"""Environment-only reward function for RL training. No expert policy —
rewards come from the game engine (score deltas) plus potential-based
shaping (distance to nearest food). Computed per legal action so the GRPO
trainer can score sampled moves with a lookup.

Reward for taking action a in state s (scaled by 1/10, clipped to [-3, 3]):
  score_delta(s, a)        engine's own scoring: +10 food, -1 time step,
                           +200 scared ghost, +500 clear, -500 death
  + 2 * (d_food(s) - d_food(s'))   shaping: 2 pts per step closer to food
  - 20 if a non-scared ghost is within 1 step of the new position
Illegal / unparseable model output gets ILLEGAL_REWARD (-1.5): worse than
any normal legal move, better than walking into a ghost.
"""

import collections

SHAPE_WEIGHT = 2.0
GHOST_ADJ_PENALTY = 40.0
GHOST_NEAR_PENALTY = 15.0
SCALE = 10.0
CLIP = 3.0
ILLEGAL_REWARD = -1.5


def bfs_dist(board, start, targets):
    """Maze distance from start to the nearest of targets, or None."""
    if not targets:
        return None
    targets = set(targets)
    seen = {start}
    queue = collections.deque([(start, 0)])
    while queue:
        pos, d = queue.popleft()
        if pos in targets:
            return d
        for _action, npos in board.get_neighbors(pos):
            if npos not in seen:
                seen.add(npos)
                queue.append((npos, d + 1))
    return None


def _food_dist(state, pos):
    return bfs_dist(state.board, pos, set(state.get_food()))


def action_rewards(state) -> dict:
    """{action string: reward} for every legal action in this state."""
    pac = state.get_agent_position(0)
    d_before = _food_dist(state, pac)
    ghosts = list(state.get_nonscared_ghost_positions().values())

    out = {}
    for action in state.get_legal_actions():
        succ = state.generate_successor(action)
        raw = succ.score - state.score

        npos = pac.apply_action(action)
        d_after = _food_dist(succ, npos)
        if d_before is not None and d_after is not None:
            raw += SHAPE_WEIGHT * (d_before - d_after)

        gdist = bfs_dist(state.board, npos, ghosts)
        if gdist is not None and gdist <= 1:
            raw -= GHOST_ADJ_PENALTY
        elif gdist == 2:
            raw -= GHOST_NEAR_PENALTY

        out[str(action)] = max(-CLIP, min(CLIP, raw / SCALE))
    return out
