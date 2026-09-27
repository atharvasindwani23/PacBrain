"""Parse GRPO training logs (Modal output) into results/training_curve.json:
one point per logged step with reward, loss, and KL — the checkpoint values
for the training graph. Usage: python parse_train_log.py <logfile>
"""

import ast
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
LOGGING_STEPS = 10


def main(path):
    points = []
    with open(path, errors="replace") as f:
        for line in f:
            m = re.search(r"\{'loss':.*?\}", line)
            if not m:
                continue
            row = ast.literal_eval(m.group(0))
            points.append({
                "step": LOGGING_STEPS * (len(points) + 1),
                "reward": row.get("reward"),
                "reward_std": row.get("reward_std"),
                "loss": row.get("loss"),
                "kl": row.get("kl"),
                "epoch": row.get("epoch"),
            })
    out = os.path.join(ROOT, "results", "training_curve.json")
    with open(out, "w") as f:
        json.dump(points, f, indent=2)
    print(f"{len(points)} checkpoints -> {out}")
    if points:
        print(f"reward: first={points[0]['reward']:.3f} "
              f"last={points[-1]['reward']:.3f}")


if __name__ == "__main__":
    main(sys.argv[1])
