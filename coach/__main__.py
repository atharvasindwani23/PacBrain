"""PacBrain coach: play -> notes in memory -> rewards and evals -> GRPO training data.

  python -m coach run    --game pacman --episodes 20            # scout, then design
  python -m coach scout  --game flappy --episodes 30 --endpoint <model>/v1
  python -m coach design --game pacman --out generated/pacman
  python -m coach states --game flappy --spec generated/flappy/reward_spec.json
  python -m coach evaluate --game flappy --endpoint <model>/v1 --label flappy_rl
  python -m coach report --game pacman --suite generated/pacman/eval_suite.json \\
      --base results/base.json --tuned results/rl.json

The coach model is any OpenAI-compatible chat endpoint (COACH_ENDPOINT, COACH_MODEL).
Notes go to GBrain unless --memory none; Pac-Man openings go through Memorable with --memorable.
"""

import argparse
import json
from pathlib import Path

from memory.service import load_env

from .designer import design, write_design
from .evals import compare, load_metrics, markdown_table
from .games import GAMES, load_game
from .llm import CoachLLM
from .memory_io import load_local_notes, publish, recall_notes
from .rewards import compile_reward, load_spec
from .scout import scout


def _scout(args, llm):
    result = scout(args.game, args.episodes, args.seed0, llm, endpoint=args.endpoint, out_dir=args.data,
                   memory=args.memory, memorable=args.memorable, max_turns=args.max_turns)
    print(json.dumps({k: v for k, v in result.items() if k != "pages"}, indent=2))
    return result


def _design(args, llm):
    memory_root = Path(args.data) / "memory"
    pages = recall_notes(args.game) if args.memory == "gbrain" else load_local_notes(memory_root, args.game)
    result = design(args.game, pages, llm)
    paths = write_design(result, args.out or f"generated/{args.game}", memory_root=memory_root)
    if args.memory == "gbrain":
        publish(memory_root)
    print(json.dumps(paths, indent=2))
    print(next_steps(args.game, paths["reward_spec"], result["reward"]["illegal_output"]))
    return result


def next_steps(game, spec_path, illegal_output):
    if game == "pacman":
        return (f"\nNext:\n"
                f"  COACH_REWARD_SPEC={spec_path} python collect_states.py 150 3000 rl_states_coach.jsonl\n"
                f"  modal run --detach modal_train_rl.py --states-file rl_states_coach.jsonl "
                f"--illegal-reward {illegal_output} --out-dir /vol/rl_coach\n"
                f"  python eval.py --endpoint <rl_coach-endpoint>/v1 --label rl_coach\n"
                f"  python -m coach report --game pacman --suite {Path(spec_path).with_name('eval_suite.json')} "
                f"--base results/base.json --tuned results/rl_coach.json")
    return (f"\nNext:\n"
            f"  python -m coach states --game {game} --spec {spec_path} --out data/{game}_states.jsonl\n"
            f"  modal run --detach modal_train_rl.py --game {game} --states-file {game}_states.jsonl "
            f"--illegal-reward {illegal_output} --out-dir /vol/{game}_rl\n"
            f"  python -m coach evaluate --game {game} --endpoint <{game}_rl-endpoint>/v1 --label {game}_rl")


def _states(args):
    if args.game == "pacman":
        raise SystemExit("Pac-Man states come from the pacai engine: "
                         f"COACH_REWARD_SPEC={args.spec} python collect_states.py")
    game = load_game(args.game)
    reward_fn = compile_reward(load_spec(args.spec, args.game), game)
    rows = game.training_states(range(args.seed0, args.seed0 + args.episodes), reward_fn, args.max_states)
    out = Path(args.out or f"data/{args.game}_states.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(row) + "\n" for row in rows))
    print(f"{len(rows)} GRPO states -> {out}")


def _evaluate(args):
    if args.game == "pacman":
        raise SystemExit("Evaluate Pac-Man with eval.py; it records GIFs and memory counters.")
    game = load_game(args.game)
    episodes = game.play_episodes(range(args.seed0, args.seed0 + args.games), endpoint=args.endpoint,
                                  label=args.label)
    out = Path("results") / f"{args.label}.json"
    if out.exists() and not args.overwrite:
        raise SystemExit(f"{out} exists; choose a new --label or pass --overwrite")
    summary = {"label": args.label, "game": args.game, "games": len(episodes),
               "metrics": game.episode_metrics(episodes),
               "per_game": [dict(e["result"], seed=e["seed"]) for e in episodes]}
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["metrics"], indent=2))


def _report(args):
    game = load_game(args.game)
    suite = json.loads(Path(args.suite).read_text())
    if suite.get("game") != args.game:
        raise SystemExit(f"{args.suite} is an eval suite for {suite.get('game')!r}")
    rows = compare(suite["evals"], load_metrics(args.base, game), load_metrics(args.tuned, game))
    print(markdown_table(rows))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m coach", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def game_arg(p):
        p.add_argument("--game", choices=sorted(GAMES), required=True)

    def memory_args(p):
        p.add_argument("--data", default="data/coach", help="Traces and memory Markdown (default data/coach)")
        p.add_argument("--memory", choices=["gbrain", "none"], default="gbrain",
                       help="Publish and recall notes through GBrain, or keep them local")

    def scout_args(p):
        p.add_argument("--episodes", type=int, default=20)
        p.add_argument("--seed0", type=int, default=5000)
        p.add_argument("--endpoint", help="Model endpoint to scout with (default: random play)")
        p.add_argument("--max-turns", type=int)
        p.add_argument("--memorable", action="store_true", help="Extract Pac-Man openings through Memorable")

    for name in ("scout", "design", "run"):
        p = sub.add_parser(name)
        game_arg(p)
        memory_args(p)
        if name != "design":
            scout_args(p)
        if name != "scout":
            p.add_argument("--out", help="Where reward_spec.json and eval_suite.json go (default generated/<game>)")

    p = sub.add_parser("states")
    game_arg(p)
    p.add_argument("--spec", required=True)
    p.add_argument("--episodes", type=int, default=400)
    p.add_argument("--seed0", type=int, default=7000)
    p.add_argument("--max-states", type=int, default=3000)
    p.add_argument("--out")

    p = sub.add_parser("evaluate")
    game_arg(p)
    p.add_argument("--endpoint", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--games", type=int, default=10)
    p.add_argument("--seed0", type=int, default=2000)
    p.add_argument("--overwrite", action="store_true")

    p = sub.add_parser("report")
    game_arg(p)
    p.add_argument("--suite", required=True)
    p.add_argument("--base", required=True)
    p.add_argument("--tuned", required=True)

    args = parser.parse_args(argv)
    load_env()
    if args.command in ("scout", "design", "run"):
        llm = CoachLLM()
        if args.command in ("scout", "run"):
            _scout(args, llm)
        if args.command in ("design", "run"):
            _design(args, llm)
    elif args.command == "states":
        _states(args)
    elif args.command == "evaluate":
        _evaluate(args)
    else:
        _report(args)


if __name__ == "__main__":
    main()
