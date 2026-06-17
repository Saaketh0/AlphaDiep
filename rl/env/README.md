# PettingZoo RL Environment Quickstart

Requires **Python 3.12+** (`../../pyproject.toml`). Older interpreters fail on modern type syntax in `rl/env/` (for example `tuple[int, ...] | None`).

This package contains the Python-side environment, rewards, actions, observation spaces, and ctypes bridge used by RL training. You should not need to browse C++ or conformance internals for normal reward experiments.

## Essential Files

`pettingzoo_env.py`
- Main PettingZoo `ParallelEnv` wrapper.
- Use `DiepCustomParallelEnv` in training scripts.
- Handles parallel agents, action dicts, observations, rewards, infos, reset, and step.

`rewards.py`
- Python reward configuration and reusable reward components.
- Important symbols: `RewardConfig`, `make_reward_config`, `reward_components`.
- Add new reusable reward fields here only when simple weights are not enough.

`actions.py`
- Converts Python trainer actions into simulator actions.
- Dict form: `{'move': [x, y], 'aim': [x, y], 'buttons': [fire, alt_fire], 'stat_upgrade_choice': i, 'tank_upgrade_choice': j}`.
- Flat form: `[move_x, move_y, aim_x, aim_y, fire, alt_fire, stat_upgrade_choice, tank_upgrade_choice]`.
- Use `-1` for either upgrade field when no upgrade is requested.

`spaces.py`
- Gymnasium/PettingZoo action and observation spaces.
- Includes tiny fallbacks for smoke tests.

`headless.py`
- Lower-level Python simulator wrapper around the C++ `diepcustom_headless_c` shared library.
- Use directly only for custom fast loops outside PettingZoo.

`agents.py`
- Profile-driven multi-agent helpers.
- Use `AgentProfile` + `AgentRoster` when each env agent needs its own build/controller config.

`__init__.py`
- Convenience exports so scripts can import from `rl.env`.

## Minimal Environment

```python
from rl.env import DiepCustomParallelEnv

env = DiepCustomParallelEnv(
    seed=1,
    agents=2,
    max_ticks=1000,
    observation_mode='combat',
    include_snapshot_info=False,
    reward_config={
        'score_delta': 1.0,
        'alive': 0.01,
        'death': -1.0,
        'step': -0.001,
    },
)
```

## Environment Contract

- Environment class: `DiepCustomParallelEnv`
- Alias: `parallel_env`
- Agent names: `agent_0`, `agent_1`, ... mapped to C ABI agent ids from `diep_agent_ids`
- Observation mode: only `combat`
- Missing live-agent actions are converted to explicit no-op actions
- Invalid or currently illegal stat/tank upgrade selections are ignored
- The wrapper does not inject AI, autopilot, scripted movement, or default firing behavior
- The C ABI episode `done` signal is treated as truncation; task-specific terminal conditions belong in trainer code

Accepted action forms:

```python
{
    "move": [move_x, move_y],
    "aim": [aim_x, aim_y],
    "buttons": [fire, alt_fire],
    "stat_upgrade_choice": 0,
    "tank_upgrade_choice": 1,
}
```

or flat sequence:

```python
[move_x, move_y, aim_x, aim_y, fire, alt_fire, stat_upgrade_choice, tank_upgrade_choice]
```

Both upgrade fields are optional and default to `-1` (no upgrade).

## Basic Loop

```python
observations, infos = env.reset(seed=1)

while env.agents:
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)
```

## Reward Fields

`RewardConfig` supports: `raw`, `score_delta`, `health_delta`, `damage_taken`, `enemy_kills`, `farm_kills`, `level_delta`, `level_milestone`, `edge_proximity`, `movement_speed`, `retreat`, `aim_accuracy`, `enemy_damage_dealt`, `alive`, `death`, `truncation`, and `step`.

For RLlib runs, tune weights with preset JSON files under `../runtime/reward_presets/` and select them from the repo root:

```bash
python rl/runtime/train.py train --reward-config path/to/file.json
```

For direct Python experiments, tune weights in your script:

```python
env.set_reward_config(score_delta=2.0, death=-2.0, step=-0.001)
```

Debug components with: `infos[agent]['reward_components']`.

## Observation Mode

Only `observation_mode='combat'` is supported.

It returns the policy-facing observation dictionary used by the RLlib combat stack:

```text
grid_obs: (18, 21, 21) float32
self_obs: (27,) float32
prev_action_obs: (5,) float32
tank_type_obs: scalar integer tank enum ID
```

The underlying C ABI writes fixed-slot batch buffers with shapes:

```text
(max_possible_agents, 18, 21, 21) float32  # combat grid, channel-first
(max_possible_agents, 27) float32          # combat self features
(max_possible_agents, 5) float32           # previous action features
```

Dead or inactive slots are zero-filled and remain present until reset. `HeadlessSim.alive_mask()` returns a parallel binary mask where `1 = alive` and `0 = terminated/inactive`.

`DiepCustomParallelEnv.env.agents` can shrink after terminations, while the underlying fixed-slot observation batch remains constant-shaped.

## RLlib Training

Production training uses Ray RLlib PPO with 20 agents: 4 mains + 16 ghosts. Use [TRAINING.md](../../TRAINING.md) as the canonical launch guide for Redis, seeding, fresh training, safe resume, and the real-Redis smoke test.

Quick commands from the repo root:

```bash
python rl/runtime/train.py seed --count 50
python rl/runtime/train.py train
python rl/runtime/train.py resume
cd ts-server && npm run test:training-smoke
```

Training data persists under `training_data/` (league weights and metadata in `redis/`, RLlib checkpoints and metadata in `RLlib/`).

## Fast Training Defaults

```python
DiepCustomParallelEnv(
    observation_mode='combat',
    include_snapshot_info=False,
    fast_reward_state=True,  # used by rl/runtime/training_runtime.py DIEP_ENV_CONFIG
    reward_config={'score_delta': 1.0, 'alive': 0.01, 'death': -1.0},
)
```

## Fast Tickless Path

For high-throughput multi-agent loops, use `HeadlessSim` directly instead of the PettingZoo dictionary API in the hot loop:

- `HeadlessSim.step_many(actions, ticks)` advances by `ticks` without per-tick Python/C crossings.
- `HeadlessSim.combat_observations_array(out=...)` fills all combat grids.
- `HeadlessSim.combat_self_observations_array(out=...)` fills compact self features.
- `HeadlessSim.combat_prev_action_observations_array(out=...)` fills previous-action features.
- `HeadlessSim.agent_states_array(out=...)` fills lightweight state rows for reward/state-vector training.
- `HeadlessSim.agent_progressions_array(out=...)` fills legal upgrade/progression state.
- `HeadlessSim.alive_mask()` exposes fixed-slot live/dead state.

The fast path is intentionally separate from reward shaping. External training code still owns rewards and terminal semantics.

## C ABI Summary

Current action struct fields (`diep_get_action_shape()` reports `9`):

1. `agent_id`
2. `move_x`
3. `move_y`
4. `aim_x`
5. `aim_y`
6. `fire`
7. `alt_fire`
8. `stat_upgrade_choice`
9. `tank_upgrade_choice`

Action layout is append-only across ABI versions. Field reinterpretation requires an ABI metadata bump.

Useful metadata calls:

- `diep_abi_version()`
- `diep_get_action_shape()`
- `diep_get_combat_observation_shape()`
- `diep_agent_ids()`
- `diep_last_error()`

Lightweight agent-state rows:

```text
(max_possible_agents, 10) float32
agent_id, alive, x, y, vx, vy, health, max_health, score, team_id
```

Progression rows:

```text
(max_possible_agents, 27) float32
level,
current_tank,
stats_available,
can_stat_upgrade,
can_tank_upgrade,
stat_0, stat_1, stat_2, stat_3, stat_4, stat_5, stat_6, stat_7,
legal_stat_0, legal_stat_1, legal_stat_2, legal_stat_3, legal_stat_4, legal_stat_5, legal_stat_6, legal_stat_7,
legal_tank_0, legal_tank_1, legal_tank_2, legal_tank_3, legal_tank_4, legal_tank_5
```

Episode stats rows:

```text
(max_possible_agents, 16) float64
lifetime_steps,
score_total,
score_from_farming,
score_from_pvp,
damage_dealt,
enemy_damage_dealt,
damage_taken,
shots_fired,
shots_hit,
enemy_kills,
farm_kills,
death_count,
death_cause,
level_reached,
tank_class,
upgrade_choices
```

The field order matches `headless.py::EPISODE_STATS_FIELDS` and is consumed by `rl/observability/core/stats_bridge.py`.

## Python Validation Files

`../../conformance/headless/python_pettingzoo_smoke.py`: quick env/reward check.

`../../conformance/headless/python_pettingzoo_api_test.py`: PettingZoo `parallel_api_test` compliance.

`../../conformance/headless/python_gym_combat_wrapper_smoke.py`: combat env smoke without external RL frameworks.

`../../conformance/headless/python_training_benchmark.py`: Python training throughput benchmark.

## Practical Workflow

1. Import `DiepCustomParallelEnv` from `rl.env`.
2. Start with `observation_mode='combat'`.
3. Tune `reward_config` in your training script.
4. Inspect `reward_components` when debugging.
5. Run smoke/API tests before long training runs.
