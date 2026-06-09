# DouZero-4A4

This folder adds a backend-compatible DouZero-style trainer without changing the
existing server, game rules, or inference path.

## Why It Fits The Current Backend

The current AI backend loads any `checkpoints/*.pt` file into
`rl.model.CardPolicyNetwork` and scores the legal actions returned by
`rl.actions.enumerate_legal_actions`. The new trainer keeps that checkpoint
format, so a trained file such as `checkpoints/douzero_4a4.pt` can be selected
by the existing API/model picker as `douzero_4a4` or `douzero_4a4.pt`.

## DouZero Adaptation

DouZero trains an action-value model with self-play Deep Monte Carlo: choose
among legal actions, wait until the game ends, then regress the selected action
value toward the final return. For 4A4:

- legal actions come from the existing 4A4 rule implementation;
- the teacher model follows DouZero's released code structure: LSTM over recent
  history, state/action concatenation, five 512-unit MLP layers, then one scalar
  Q-value per legal action;
- the target is terminal team utility, not per-step shaping;
- teammates receive identical rewards;
- half/full hole, special levels 3/J/A, and stage change are included in the
  terminal utility.

The existing backend loader always instantiates `rl.model.CardPolicyNetwork`.
To keep server code untouched, every checkpoint stores:

- `douzero_model_state_dict`: the real DouZero-style teacher used for self-play;
- `model_state_dict`: a backend-compatible student trained from the same DMC
  targets, which the current server can load directly.

No intermediate reward is given for using bombs, cha/dian, passing, finishing
first alone, or reducing hand size. Those shortcuts can easily teach the model
to burn important cards, block its teammate, or optimize solo rank instead of
team victory.

## Remote Training

On `dev.dyeea.com:17021`, use the `torch` conda environment:

```bash
conda activate torch
cd /path/to/4A4
python scripts/train_douzero_4a4.py \
  --frames 1000000 \
  --device cuda \
  --num-actors 4 \
  --collect-episodes 8 \
  --batch-size 256 \
  --update-frames 8192 \
  --save checkpoints/douzero_4a4.pt \
  --checkpoint-every 50000 \
  --eval-every 50000 \
  --log-jsonl logs/douzero_4a4_metrics.jsonl
```

The trainer writes periodic checkpoints next to the final checkpoint and logs
JSONL metrics that can be tailed during training.
