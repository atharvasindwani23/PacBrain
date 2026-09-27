# PacBrain: fine-tuned DeepSeek plays Pacman

Same 1.5B DeepSeek model, before and after an 8-minute LoRA fine-tune
on 23k expert moves. Identical prompts, seeds, and board.

| model | games | avg score | win rate | illegal moves |
|---|---|---|---|---|
| base | 10 | -428 | 0% | 82.5% |
| tuned | 10 | 387 | 50% | 0.0% |

## Before (base model)
![base_seed2000.gif](base_seed2000.gif)
![base_seed2001.gif](base_seed2001.gif)

## After (fine-tuned)
![tuned_seed2000.gif](tuned_seed2000.gif)
![tuned_seed2001.gif](tuned_seed2001.gif)
