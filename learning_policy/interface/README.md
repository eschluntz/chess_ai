# Head-to-Head Arena

Web UI + CLI to pit two engines against each other:
- policy checkpoint vs policy checkpoint
- policy checkpoint vs classical negamax (`core/search.py`)

The UI supports:
- play/pause
- step one move at a time
- max-speed autoplay
- per-side `temp=0` / `temp=1`
- move arrows for probabilities `>= 20%`
- multi-game series summaries (win/loss/draw rates)
- built-in common opening selector (ECO-tagged) for single-game inspection
- diverse-opening series mode (all common openings, two games each with color swap)

Note: `Run Match` initializes a live session only. Moves are computed incrementally when
you press `Play` or `Step +1`.

## Run Web UI

From `learning_policy/`:

```bash
python interface/server.py \
  --vocab cache/planes/vocab.npy \
  --host 127.0.0.1 \
  --port 5000
```

Then open:

`http://127.0.0.1:5000`

## Run CLI Series

From `learning_policy/`:

```bash
python interface/head_to_head.py \
  --vocab cache/planes/vocab.npy \
  --white-kind policy \
  --white-checkpoint checkpoints/scale-20M.pt \
  --white-temp 0 \
  --black-kind policy \
  --black-checkpoint checkpoints/scale-40M.pt \
  --black-temp 0 \
  --games 50 \
  --swap-colors
```

Policy vs classical:

```bash
python interface/head_to_head.py \
  --vocab cache/planes/vocab.npy \
  --white-kind policy \
  --white-checkpoint checkpoints/scale-20M.pt \
  --black-kind negamax \
  --black-eval piece_position \
  --black-depth 2 \
  --games 20 \
  --swap-colors
```

## Download Example Checkpoints

Recent run names from `experiments.md` include:
- `scale-10M`
- `scale-20M`
- `scale-40M`
- `scale-80M`
- `scale-160M`

Example commands (Modal volume `chess-policy-checkpoints`):

```bash
mkdir -p checkpoints

modal volume get chess-policy-checkpoints checkpoints/scale-20M.pt checkpoints/scale-20M.pt
modal volume get chess-policy-checkpoints checkpoints/scale-40M.pt checkpoints/scale-40M.pt
```

If your Modal CLI uses a different subcommand shape, run `modal volume --help` and pull these same paths from the `chess-policy-checkpoints` volume.
