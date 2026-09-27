#!/bin/sh
# Live PacBrain demo: watch the model play in your browser.
#   ./demo.sh base            # the "before" — untuned DeepSeek
#   ./demo.sh rl              # the "after" — trained from rewards only (GRPO)
#   ./demo.sh tuned           # older SFT checkpoint
#   ./demo.sh both            # side-by-side: base vs rl, two tabs, same seed
# Optional second arg: seed (default 2000).

cd "$(dirname "$0")" || exit 1
MODEL="${1:-rl}"
SEED="${2:-2000}"
BASE_URL="https://pranav100000--pacbrain-serve-serve-base.modal.run/v1"
TUNED_URL="https://pranav100000--pacbrain-serve-serve-tuned.modal.run/v1"
RL_URL="https://pranav100000--pacbrain-serve-serve-rl.modal.run/v1"

warm() {
  echo "Warming $1 endpoint (cold start can take ~2 min)..."
  until curl -s -o /dev/null --max-time 10 "$1/models"; do
    printf "."; sleep 5
  done
  echo " ready."
}

play() { # $1=endpoint url
  PACBRAIN_ENDPOINT="$1" .venv/bin/python -m pacai.pacman \
    --ui web --fps 6 --board classic-small \
    --pacman llm_agent.py:LLMAgent --seed "$SEED" --max-turns 600
}

case "$MODEL" in
  base)  warm "$BASE_URL";  play "$BASE_URL" ;;
  tuned) warm "$TUNED_URL"; play "$TUNED_URL" ;;
  rl)    warm "$RL_URL";    play "$RL_URL" ;;
  both)
    warm "$BASE_URL"; warm "$RL_URL"
    play "$BASE_URL" &
    sleep 3
    play "$RL_URL"
    wait ;;
  *) echo "usage: ./demo.sh [base|tuned|rl|both] [seed]"; exit 1 ;;
esac
