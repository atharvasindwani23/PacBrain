"""Single source of truth: GameState -> text prompt, and model text -> action.
Used by BOTH data generation (expert) and inference (LLM agent). Never format
prompts anywhere else.
"""

import re

import pacai.core.action
import pacai.pacman.board

ACTIONS = {
    "north": pacai.core.action.NORTH,
    "south": pacai.core.action.SOUTH,
    "east": pacai.core.action.EAST,
    "west": pacai.core.action.WEST,
    "stop": pacai.core.action.STOP,
}

PROMPT_HEADER = (
    "You are Pacman. Grid: % wall, . food, o capsule, G ghost, g scared ghost, "
    "P you. Reply with exactly one move word.\n"
)


def serialize(state) -> str:
    board = state.board
    walls = board.get_walls()
    height, width = board.height, board.width

    grid = [[" "] * width for _ in range(height)]
    for p in walls:
        grid[p.row][p.col] = "%"
    for p in state.get_food():
        grid[p.row][p.col] = "."
    for p in board.get_marker_positions(pacai.pacman.board.MARKER_CAPSULE):
        grid[p.row][p.col] = "o"
    for idx, p in state.get_nonscared_ghost_positions().items():
        grid[p.row][p.col] = "G"
    for idx, p in state.get_scared_ghost_positions().items():
        grid[p.row][p.col] = "g"
    pac = state.get_agent_position(0)
    if pac is not None:
        grid[pac.row][pac.col] = "P"

    # A winning final state has no active agent. The terminal observation is
    # still useful to memory, but requesting its next legal action raises.
    legal = [] if state.game_over else [str(a) for a in state.get_legal_actions()]
    lines = ["".join(row) for row in grid]
    return (
        PROMPT_HEADER
        + "\n".join(lines)
        + "\nLegal moves: " + ", ".join(legal)
        + "\nMove:"
    )


def parse_action(text: str, legal_actions) -> "pacai.core.action.Action | None":
    """Map model output text to a legal action, or None."""
    if not isinstance(text, str):
        return None
    # The contract is one move word: do not turn "northeast", a negated
    # direction, or a reasoning paragraph into an apparently legal decision.
    match = re.fullmatch(r"\s*(north|south|east|west|stop)[.!]?\s*", text, re.IGNORECASE)
    if not match:
        return None
    legal = {str(a).lower(): a for a in legal_actions}
    return legal.get(match.group(1).lower())
