"""PacBrain coach: the half of the pipeline that comes before training.

The coach plays a game many times, writes grounded notes about each episode
into memory (GBrain pages, with Pac-Man openings extracted through
Memorable), then reads those notes back to design the reward function and
eval suite that the Modal GRPO trainer (modal_train_rl.py) consumes.
"""
