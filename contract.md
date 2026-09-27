# Gameplay ↔ memory contract

The game runner owns observation creation, River inference, and action execution. Memory proposes **one action at a time** from a previously verified opening. The game always supplies a fresh observation before the next action.

## Trace input

Use [examples/synthetic_trace.json](examples/synthetic_trace.json) as a format fixture, not as evidence of gameplay success.

| Field | Required meaning |
|---|---|
| `episode_id` | Nonempty stable identifier for this run |
| `layout` | Nonempty exact layout name, shared with recall |
| `observation_timing` | Set explicitly to `"pre_action"` |
| `seed` | Record the actual seed when available |
| `frames` | Ordered observations immediately before each executed action |
| `frames[].grid` | Rectangular ASCII string with newlines, or array of equal-length row strings |
| `frames[].score` | Finite numeric score before this frame's action |
| `frames[].action` | The action **actually executed**: `N`, `S`, `E`, or `W` |
| `frames[].source` | `policy` for the policy's executed choice; `procedure` for memory; use a different value such as `fallback` for corrections/random actions so extraction stops there |
| `frames[].action_ok` | `true` or `false` only when the environment knows the action outcome; omit if unknown. This is tool execution success, not winning the game |
| `frames[].legal_moves` | Optional actual legal actions; when present, memory intersects them with grid legality |
| `frames[].died` | Record `true` when the observation/action carries a known death |
| `terminal_observation` | Optional observation after the final action, including `grid`, `score`, and known death state; needed to verify that final transition |
| `result` | Outcome object; include actual `score`, `cleared`, `died`, `turns`, and observed call counts when available |

Grid glyphs are `%` wall, `.` pellet, `o` power pellet, `P` Pac-Man, `G` active ghost, `g` scared ghost, and space for empty floor. Exactly one `P` is required for an actionable observation. Coordinates derived from this grid are zero-based `(x, y)`, with x increasing right and y increasing down. Optional `pacman`, `ghosts`, and `scared` fields are metadata; memory checks the grid itself. Do not leave scared ghosts rendered as `G`.

A frame must contain the board **before** its action. The next frame confirms where that move actually landed and the resulting score. Do not record a whole proposed action batch as executed. Log each action individually, including corrections under their actual source. Mark a synthetic fixture explicitly with `synthetic: true`.

## What becomes a procedure

Extraction takes a contiguous opening, bounded by `max_steps` (default 40), with positive total observed reward. Each move must match the next observed position, stay in the same wall topology, be legal, and begin beyond the configured active-ghost distance (default 2 maze-path steps). Individual movement costs of −1 are allowed. Extraction stops at the first unsupported transition, fallback action, death, danger, or missing observation.

The Memorable draft must contain exactly the independently validated action sequence. An invented, reordered, or missing action rejects the draft. Stored JSON includes the layout, wall-topology hash, episode/seed provenance, observed transitions, reward, extraction method, and provider receipt. Observed reward is evidence from that opening, not a predicted success rate.

## Per-episode replay

```python
from memory.core import ProcedureCursor
from memory.service import load_env, recall_procedure

load_env()
hit = recall_procedure(layout, initial_observation, provider="gbrain")
cursor = ProcedureCursor(hit["procedure"]) if hit["procedure"] else None

# Inside the game loop, before each individual env.step:
action = cursor.next_action(observation, layout) if cursor else None
fallback_reason = None if action is not None else (
    cursor.reason if cursor else "no_matching_procedure"
)
if action is None:
    action = policy_choose_action(observation)  # teammate's River integration
    source = "policy"
else:
    source = "procedure"
# Call env.step(action) now, then record the actual transition and source.
```

These game-loop names are integration placeholders. Call `next_action` exactly once for each immediately attempted move: it advances the cursor when it returns an action. Do not call it for previews or advance it while waiting for inference. Create a new cursor per episode. If the environment rejects an action, record the failed outcome and stop using that cursor.

The cursor stops permanently on `complete`, `layout_changed`, `topology_changed`, `position_diverged`, `illegal_action`, `ghost_nearby`, or `invalid_observation_or_procedure`. Its successful selection reason is `recalled_step`. Once stopped, call the policy on the current observation. Provider errors should be surfaced separately as `memory_unavailable`; an intentional local fallback must be labeled `provider: local`.

For the viewer, record `source`, procedure ID, retrieval provider, cursor step, and `fallback_reason` per decision. Count actual policy calls and actual procedure actions, so a demo can show saved calls without inventing a performance gain.
