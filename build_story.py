"""Inject eval + training data into results/pacbrain_story.html between the
/*DATA*/ ... /*/DATA*/ markers. Run after eval.py and parse of log history."""

import json
import os
import re

ROOT = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(ROOT, "results")


def main():
    data = {
        "base": json.load(open(os.path.join(R, "base.json"))),
        "rl": json.load(open(os.path.join(R, "rl.json"))),
        "curve": json.load(open(os.path.join(R, "training_curve.json"))),
    }
    hist = json.load(open(os.path.join(R, "log_history.json")))
    tr = [r for r in hist if "train_runtime" in r]
    data["train_minutes"] = round(tr[0]["train_runtime"] / 60) if tr else 11

    path = os.path.join(R, "pacbrain_story.html")
    html = open(path).read()
    html = re.sub(r"/\*DATA\*/.*?/\*/DATA\*/",
                  "/*DATA*/" + json.dumps(data) + "/*/DATA*/",
                  html, flags=re.S)
    open(path, "w").write(html)
    print("injected: base games", data["base"]["games"],
          "| rl games", data["rl"]["games"],
          "| curve points", len(data["curve"]))


if __name__ == "__main__":
    main()
