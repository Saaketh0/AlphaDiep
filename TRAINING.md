# Diep RL Training

This is the short operator guide for starting training, testing it, viewing W&B/videos, and managing the Redis/league data stores.

## Setup

Setting up the env to train
```bash
uv run rl/runtime/train.py doctor
```

## Training

Need 50 past copies of agents for self-play league play, hence the first command
```bash
uv run rl/runtime/train.py seed --count 50 # run only if the redis cache is empty (usually only once initially)
uv run rl/runtime/train.py train # start training
uv run rl/runtime/train.py resume # continue latest run
```

## Tests

For Python RL changes, use `pytest` first:

```bash
uv run pytest rl/runtime/testing
uv run pytest rl/observability/tests
uv run conformance/headless/python_pettingzoo_smoke.py
```

For the full mixed-language repo checks, also run:

```bash
(cd ts-server && npm run test:all)
(cd ts-server && npm run test:training-smoke)
```

## Redis and League Data

Redis holds live ghost-league weights. SSD exports under `training_data/redis/` hydrate Redis when memory is empty.

```bash
uv run rl/runtime/train.py redis start
uv run rl/runtime/train.py redis status
uv run rl/runtime/train.py redis stop
```

Artifact layout:

| Path | Purpose |
| --- | --- |
| `training_data/redis-server/` | Redis AOF bind mount |
| `training_data/redis/` | Lean league exports and `league_metadata.json` |
| `training_data/RLlib/` | RLlib/Tune checkpoints and run metadata |
| `training_data/W&B/` | W&B logs, observability output, and videos |

The Redis container uses AOF with RDB snapshots disabled. RLlib checkpoints exclude ghost modules and optimizer state; ghosts reload from the league track on init/resume.

## W&B

Full training writes W&B-compatible metrics by default. The default mode is offline-first:

```bash
WANDB_MODE=offline uv run rl/runtime/train.py train
```

Local W&B runs live under:

```bash
training_data/W&B/wandb/
```

Sync runs after training:

```bash
uv run wandb sync training_data/W&B/wandb/offline-run-*
```


Create W&B reports in the web UI after syncing or while online training is running.

## Videos

One-off video without training:

```bash
uv run rl/runtime/generate_training_video.py \
  --run-id video-observation \
  --scenario training-ffa-easy \
  --agents 20 \
  --max-ticks 300 \
  --max-steps 300 \
  --video-agent agent_1 \
  --seed 7 \
  --fps 20
```

Periodic videos during training:

```bash
DIEP_VIDEO_ENABLED=true \
DIEP_VIDEO_AGENT=agent_1 \
DIEP_VIDEO_INTERVAL=500 \
DIEP_EVAL_MAX_STEPS=1000 \
uv run rl/runtime/train.py train
```

Videos are saved under `training_data/W&B/videos/` and attached to W&B as `gameplay/training_video`. Video failures are logged but should not stop training.

## Notes

- Ghost lineup is global: all parallel envs share the same 16 ghost policy IDs at any instant.
- RLlib checkpoints do not restore mid-episode C++ simulator state; env runners restart episodes on resume.
- Smoke runs disable W&B and keep checkpoint retention small.
