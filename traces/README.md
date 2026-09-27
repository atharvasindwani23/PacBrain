# Game traces (seam for Memorable / GBrain)

One JSON file per game: `expert_<seed>.json`

```json
{
  "game": "pacman",
  "board": "classic-small",
  "seed": 1000,
  "agent": "expert",
  "score": 967,
  "win": true,
  "steps": [
    {"prompt": "<serialized board state>", "completion": " North"},
    ...
  ]
}
```

- `steps[i].prompt` is the exact text the model sees (ASCII board + legal moves).
- `steps[i].completion` is the move taken.
- Post a trace to Memorable's `POST /v1/extract` as a JSON tool-trace, or index
  the (score, win, steps) summaries into GBrain as markdown notes.
