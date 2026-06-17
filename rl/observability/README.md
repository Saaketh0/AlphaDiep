# Diep.io RL Observability

Observability for headless DiepCustom RL training. The supported training path is Ray Tune's `WandbLoggerCallback` plus `DiepRLlibObservabilityCallback`. Legacy custom `WandbLogger` / `DiepMetricsCallback` wiring has been removed.

| Deliverable | What you get | When |
| --- | --- | --- |
| **Logged stats** | W&B offline-first curves and four per-main-tank iteration tables | Every real train/resume run — “is learning / reward shaping working?” |
| **Watchable playback** | Periodic MP4 attached to the same W&B run as `gameplay/training_video` | Every `DIEP_VIDEO_INTERVAL` training iterations when video is enabled |

Training stays fast: only cheap C++ counters run every step. Video is sampled periodically and failures are non-fatal. Smoke runs remain W&B-free.

**Stack context:** C++ sim → ctypes C ABI → PettingZoo → Ray RLlib (`rl/runtime/ray_code.py`). This package extends that stack; it does not replace the sim.

---

## Architecture

```text
                 ┌──────────────────────────────────────┐
                 │  core — C++ EpisodeStats (always on) │
                 │  + combat obs (eval overlays only)   │
                 └──────────────────┬───────────────────┘
                                    │
            ┌───────────────────────┴───────────────────────┐
            ▼                                               ▼
   logging pipeline                              video pipeline
   (real train/resume runs)                      (periodic training sample)
            │                                               │
   DiepRLlibObservabilityCallback                FFmpeg renderer
   + Ray WandbLoggerCallback                     → `gameplay/training_video`
   → W&B scalars + per-agent tables
```

Both pipelines read the **same** `EpisodeStats` and (for video) the same combat observation the policy sees.

---

## Package layout

```text
diepcustom/rl/observability/
  README.md                 # this file
  __init__.py
  config.py                 # ObservabilityConfig (wandb mode, agents, paths)
  requirements-extras.txt

  core/
    metrics_schema.py       # field names, W&B keys, EpisodeStats C++ layout
    observation_schema.py   # combat obs re-exports + channel helpers
    stats_bridge.py         # C++ buffer → EpisodeStatsSummary dataclass

  logging/
    wandb_tune.py           # Ray Tune WandbLoggerCallback kwargs/factory
    rllib_callbacks.py      # main-tank metrics, W&B tables, periodic video metadata

  video/
    render_grid_obs.py      # grid_obs channel composites
    render_overlay.py       # trajectory, aim, on-screen stat text
    video_writer.py         # FFmpeg encode, bounded queue

  tests/
    test_episode_stats.py
    test_reward_components.py   # integration: env + reward_config + logging
    test_observation_shapes.py

  runs/
    benchmarks.md
    <run_id>/
      episodes.jsonl          # one JSON line per episode (logging fallback)
      eval/
        <episode_id>/
          eval.mp4
          episode_summary.json
```

**Outside this package (thin wiring only):**

| Path | Role |
| --- | --- |
| `diepcustom/cpp/` | `EpisodeStats` counters + C ABI export (current ABI v10) |
| `diepcustom/rl/env/headless.py` | ctypes + `episode_stats_array()` |
| `diepcustom/rl/env/pettingzoo_env.py` | Combat obs; optional `enable_episode_stats` later |
| `diepcustom/rl/env/observations/combat.py` | Schema source of truth for `grid_obs` channels |
| `diepcustom/rl/runtime/ray_code.py` | RLlib PPO training entry with metadata-bound W&B wiring |
| `diepcustom/rl/runtime/ghost_model.md` | Ghost league loop, Redis/SSD persistence, resume requirements |
| `diepcustom/rl/observability/` | Eval video pipeline entry point (planned) |

**Imports** (add `diepcustom/` to `PYTHONPATH`):

```python
from rl.observability.config import ObservabilityConfig
from rl.observability.logging.rllib_callbacks import DiepRLlibObservabilityCallback
from rl.observability.logging.wandb_tune import create_wandb_logger_callback
from rl.observability.core.stats_bridge import EpisodeStatsSummary
```

---

## Module: `core`

### Purpose

Per-agent combat counters updated inside the C++ sim hot loop. One ABI read per agent per episode boundary — never per step.

### C++ `EpisodeStats` (per slot, reset on `diep_reset`)

| Field | Meaning |
| --- | --- |
| `lifetime_steps` | Ticks alive |
| `score_total` | Final score |
| `score_from_farming` | Score from shape kills |
| `score_from_pvp` | Score from agent kills |
| `damage_dealt`, `damage_taken` | Combat damage totals |
| `shots_fired`, `shots_hit` | Fire and hit counts |
| `kills`, `death_count` | Kill count; 0/1 died this episode |
| `death_cause` | Enum: projectile, collision, boundary, unknown, … |
| `level_reached`, `tank_class` | Progression at flush |
| `upgrade_choices` | Packed stat/tank upgrades applied |

**Hook points in `headless.cpp`:** `fireProjectile`, `receiveDamage`, kill branch, `tryApplyStatUpgrade`, `tryApplyTankUpgradeSlot`.

**ABI:** current `diep_abi_version()` is **10**; episode stats use `diep_episode_stats_fields()` and `diep_episode_stats(sim, buf, len)`.

### Python `EpisodeStatsSummary`

Wraps C++ row + callback-only fields:

- `episode_id`, `controlled_agent`, `total_reward` (rollout sum)
- `hit_rate` = `shots_hit / max(1, shots_fired)`
- `farm_vs_pvp_ratio` = farm / (farm + pvp)

`stats_bridge.py` + `metrics_schema.py` own field names and W&B keys.

---

## Module: `logging`

### Purpose

Log combat outcomes and reward-shaping signals across **many episodes** so you can compare runs — especially when tuning `reward_config`.

### `DiepRLlibObservabilityCallback`

Reads `episode_stats_array()` on episode end and accumulates raw per-step `info['reward_components']` for the four main tanks: `agent_0`, `agent_1`, `agent_2`, and `agent_3`. It logs raw `*_sum` reward components only; normalized reward metrics and per-step normalized means are intentionally omitted.

Each training iteration appends one row per main tank to W&B Tables logged as `main/agent_0/iteration_table` through `main/agent_3/iteration_table`.

### W&B metrics (primary dashboard)

**Training**

| Key | Source |
| --- | --- |
| `main/agent_N/reward/<component>_sum` | Raw reward component episode sum |
| `main/agent_N/game/score_total` | Native episode score |
| `train/policy_entropy` | trainer logger (when attached) |
| `train/explained_variance` | trainer logger (when attached) |

**Combat** (from `EpisodeStats`)

| Key | Source |
| --- | --- |
| `main/agent_N/game/hit_rate` | Derived |
| `main/agent_N/game/enemy_kills` | `enemy_kills` |
| `main/agent_N/game/level_reached` | `level_reached` |
| `main/agent_N/game/score_from_farming` | `score_from_farming` |
| `main/agent_N/game/score_from_pvp` | `score_from_pvp` |
| `main/agent_N/game/damage_dealt`, `main/agent_N/game/damage_taken` | C++ counters |
| `main/agent_N/game/death_cause` | Categorical |

**Reward shaping** (from `reward_components` in `info`, requires `reward_config` on env)

| Key | Source |
| --- | --- |
| `main/agent_N/reward/score_delta_sum` | Raw per-episode aggregate |
| `main/agent_N/reward/damage_taken_sum`, `main/agent_N/reward/death_sum`, … | One raw sum per `RewardConfig` field |

**Environment**

| Key | Source |
| --- | --- |
| `env/steps_per_second` | Wall clock |

### Run config logging

Run identity is canonicalized through `TrainingRunMetadata.run_id`: real train/resume uses that value as W&B `id` and a `run_id:<id>` tag. The human-facing W&B name is `run-<run_id>-<month>-<day>` (for example `run-run-abc123-06-15`). The Tune experiment name (for example `rl_run`) is the W&B `group`. On resume, `run_metadata.json` is read before restore and the same W&B run id is reused. Checkpoints are not uploaded to W&B by default; only checkpoint path/metadata is logged.

### CLI/environment

| Setting | Default | Meaning |
| --- | --- | --- |
| `WANDB_MODE` | `offline` | Offline-first W&B logging for real train/resume |
| `WANDB_UPLOAD_CHECKPOINTS` | off | Keep off unless W&B becomes the restore/artifact store |
| `DIEP_OBS_STEP_INTERVAL` | `1` | Sample reward components every K env steps |
| `DIEP_VIDEO_AGENT` | `agent_1` | Agent camera followed by training video |

C++ maintains counters for **all** slots; the RLlib callback logs the four main tanks.

### Episode boundary

Flush stats when the **controlled agent** `terminated` or `truncated`, or on `reset()`. Global C++ `done` (all dead or max ticks) is not the only signal — learner death can end the logged episode while other agents may still be alive.

---

## Module: `video`

### Purpose

Make stats and perception **visible** for one episode. Primary human debug tool; complements W&B curves.

### Enabling video during training

Training video is **off by default** for `ray_code.py` because the rollout + per-frame `env.snapshot()` render is synchronous. Opt in with `DIEP_VIDEO_ENABLED=true` (also honors `DIEP_VIDEO_INTERVAL`, default `500`, `DIEP_VIDEO_FPS`, and `DIEP_VIDEO_AGENT`, default `agent_1`). The video is a direct full-world gameplay render centered on the selected agent, not the policy `grid_obs` screen, so nearby tanks, shapes, crashers, projectiles, arena bounds, health bars, and heading lines are visible.

### Frame content

**From full-world snapshots** (`env.snapshot()`):

- Agents/tanks, shapes/farmables, crashers, projectiles, and arena bounds
- Camera centered on `DIEP_VIDEO_AGENT` (default `agent_1`)
- Health bars, agent labels, heading/barrel lines, and selected-agent camera label

**Overlays**

- Position trail, aim vector, fire indicator (from `prev_action_obs` — current-tick applied action; see below)
- On-screen text: tick, health, level, step reward, cumulative reward
- Running totals: farm/pvp score, hit rate (from same C++ counters as logging)

### Outputs per training-video iteration

```text
training_data/W&B/videos/
  DD-MM-iteration-<N>.mp4
  DD-MM-manual-testing.mp4
```

Videos are written under `training_data/W&B/videos/`. Automatic training videos use `DD-MM-iteration-<N>.mp4`; standalone observation videos use `DD-MM-manual-testing.mp4`. Upload the MP4 to W&B as `gameplay/training_video` when W&B is enabled.

**Speed rule:** bounded encode queue; drop frames if FFmpeg falls behind. Zero impact on training throughput.

---

## Reward function optimization workflow

This plan is designed for iterating on `reward_config` (see `rl/env/rewards.py` and `DiepCustomParallelEnv`).

```text
1. Set reward_config on env
2. Train with logging on (W&B offline or JSONL)
3. Compare runs on:
   - train/episode_reward
   - `main/agent_N/reward/*_sum` component sums
   - `main/agent_N/game/score_from_farming` vs `score_from_pvp`, hit_rate, episode_length
4. Shortlist runs that look best OR suspicious (high reward, low survival, etc.)
5. eval_with_visuals.py on those checkpoints → MP4 + episode_summary.json
6. Adjust weights; repeat
```

**Why both pipelines matter:** shaped reward can rise while behavior worsens (reward hacking). Logging `game/*` alongside `reward/*` catches that; video explains *how* on a single episode.

---

## Combat observation (shared with policy)

Video overlays and future health checks use the **same** combat dict as the RLlib `DiepPolicy` encoder:

| Key | Shape | Notes |
| --- | --- | --- |
| `grid_obs` | `(18, 21, 21)` | Channel-first float32 |
| `self_obs` | `(27,)` | Health, level, movement, stats, derived stats, recent damage |
| `prev_action_obs` | `(5,)` | Gym key name; holds **current-tick** applied action |
| `tank_type_obs` | scalar | Exact current tank enum ID; unknown/default fallback is 56 |
| `applied_action_obs` | `(5,)` | Optional alias (same buffer as `prev_action_obs`) |

Schema: `rl/env/observations/combat.py`. Helpers: `core/observation_schema.py`.

Auxiliary buffers for overlays (not policy input): `agent_states_array` `(agents, 10)` for world x/y.

---

## `ObservabilityConfig`

Single config object for trainers and eval script:

```python
@dataclass
class ObservabilityConfig:
    enabled: bool = True
    wandb_mode: str = "offline"       # off | offline | online
    stats_log_agents: tuple[str, ...] = ("agent_0",)
    run_id: str | None = None
    runs_dir: Path = Path("rl/observability/runs")
```

Video pipeline reads the same config for output paths; it does not run during training.

---

## Risks & implementation notes

Guardrails for Milestones A–C. Read this before touching C++ counters, the callback, or the video writer.

### Parallel environments (VecEnv)

**Today:** RLlib training uses parallel env runners; observability must remain episode-boundary safe under concurrency.

**When vectorizing**, each sub-environment must own an isolated C++ handle:

| VecEnv type | Requirement |
| --- | --- |
| `DummyVecEnv` | N env objects in one process → N separate `HeadlessSim` instances, each with its own `diep_sim*` |
| `SubprocVecEnv` | One `diep_sim*` per subprocess; prefer **spawn** over fork on macOS if dylib state causes issues |

**VecEnv checklist**

- [ ] One `diep_create` / `diep_destroy` pair per sub-env — never share a `HeadlessSim` across workers
- [ ] `EpisodeStats` buffer lives **inside** `Simulation` (or `diep_sim`), not in static/global memory
- [ ] Callback logs per-env stats keyed by sub-env id when VecEnv is enabled
- [ ] No cross-env reads of `_sim` without indexing through `VecEnv.get_attr` / per-env unwrapped handles

### C++ ABI memory isolation (current model)

Isolation is **per opaque `diep_sim*` handle**, not process-global:

```text
diep_create()  →  new diep_sim { unique_ptr<Simulation>, rewards, snapshot }
diep_step(sim, …) / diep_agent_states(sim, …)  →  all state on that handle
HeadlessSim.handle  →  one ctypes pointer per Python env wrapper
```

`load_library()` caches the shared `.dylib` in `_LIB`; that is the library loader only — sim state stays on each handle. **EpisodeStats (v9)** must follow the same pattern: fixed array per sim, exported via `diep_episode_stats(sim, buf, len)` like `diep_agent_states`.

**C++ memory management**

- Pre-allocate one `EpisodeStats` struct (or `[max_agents]` row array) per sim at `diep_create` / `Simulation` construction
- Zero counters on `diep_reset` with `memset` or field-wise clear — **no per-episode heap allocation**
- Avoid fragmentation and GC-adjacent pauses from churning temporary buffers in the hot loop

### Reward logic (no duplication)

| Layer | Responsibility |
| --- | --- |
| C++ | Raw counters only (`shots_fired`, `damage_dealt`, `score_from_farming`, …) |
| Python env | Per-step `info['reward_components']` from `reward_config` (`rewards.py`) |
| Callback | Raw episode **sums** of `reward_components` at flush — never mirror `RewardConfig` in C++ |

If reward weights live in Python but aggregates are computed in C++, configs will drift. **Do not** add shaped reward math to `headless.cpp` for observability.

### Callback I/O (avoid training stalls)

`DiepRLlibObservabilityCallback` must not block the env step loop:

- **No network or disk I/O on every `step()`** — accumulate `reward_components` in memory during the episode
- **Flush at episode boundary only** (controlled agent `terminated` / `truncated`, or `reset()`)
- **Batch** one `wandb.log` payload per episode (or per rollout), not per scalar per step
- Initialize W&B with background sync where possible (`wandb.init` + default async upload); JSONL append is also episode-boundary only

With a single env and episode-boundary logging, C++ throughput should stay within the **< 2%** regression target. Per-step W&B scalars for each reward component are explicitly out of scope.

### Video pipeline optimizations

Periodic and failure-tolerant; no hot-loop network or disk writes.

- **FFmpeg stdin piping:** pipe raw RGB NumPy frames into an `ffmpeg` subprocess stdin instead of OpenCV `VideoWriter` or intermediate frame files — OpenCV heatmaps are CPU-bound; avoid double encode paths
- **Bounded queue:** if FFmpeg falls behind, **drop frames** rather than blocking the eval loop (already the speed rule in `video/`)
- Prefer subprocess encode over writing PNG sequences to disk

### Optional / deferred metrics

**Action oscillation** (direction sign-flip rate per slot) is a cheaper thrashing signal than per-step policy entropy. Not in MVP. Options when needed:

- Derive coarse oscillation from `prev_action_obs` in **eval only** (Python)
- Add a single counter in C++ on direction/aim sign changes (ABI v9+ or later)

---

## Implementation milestones

```text
Milestone A — core
  C++ EpisodeStats + ABI v9
  headless.py episode_stats_array()
  core/metrics_schema.py, core/stats_bridge.py
  tests/test_episode_stats.py

Milestone B — logging
  logging/wandb_tune.py
  logging/rllib_callbacks.py
  Wire train_rppo_vs_dummy_bots.py (CLI flags)
  tests/test_reward_components.py

Milestone C — video
  video/render_*.py, video/video_writer.py
  training video render attached as gameplay/training_video
  Optional W&B video upload

Milestone D — glue (optional)
  Upload training video artifacts linked to metadata run id
  enable_episode_stats on pettingzoo_env (lazy import)
  applied_action_obs alias in observation dict

Deferred (separate PRs, not blocking MVP):
  Full step traces (old “Layer 2”), Parquet, replay loader
  Automated failure sampler (use manual eval first)
  Zero-copy observations (after health checks; pybind11 only if <20% FPS gain)
  C++ DebugEventBuffer, Tracy/perf profiling
```

### MVP success

1. Train with logging → W&B offline or `episodes.jsonl` shows reward, combat stats, and reward component aggregates.
2. Run eval script → MP4 shows grid + overlays; `episode_summary.json` matches logged stats for that episode.
3. `python_training_benchmark.py` shows **< 2%** regression with logging on (video not measured during training).

### Baseline benchmark (before and after each milestone)

```bash
cd diepcustom
npm run test:cpp
uv run python conformance/headless/python_training_benchmark.py
cd rl/runtime && ./start_redis.sh
PYTHONPATH=.. uv run python -m league_initialization.seed_league_cache   # first time only
PYTHONPATH=.. uv run python ray_code.py
```

Before `ray_code.py`, start Redis (`rl/runtime/start_redis.sh`) and seed the ghost league once. See [rl/runtime/ghost_model.md](../rl/runtime/ghost_model.md).

Record results in `rl/observability/runs/benchmarks.md` (date, git SHA, ticks/sec, training wall-clock).

### Dependencies

Use the repo-level UV environment:

```bash
uv sync
```

MVP: `wandb`, `pytest`. Add future optional video/image packages to `pyproject.toml` before syncing.

---

## Locked decisions

| Topic | Decision |
| --- | --- |
| Package path | `diepcustom/rl/observability/` |
| Action obs | Keep `prev_action_obs` for RLlib; add `applied_action_obs` alias when Milestone D lands |
| Env hook | Optional later (`enable_episode_stats`); v1 callback reads `_sim` directly |
| Traces | No full step traces in MVP; video + episode summary instead |
| pybind11 | Only if zero-copy ctypes misses **≥ 20%** env FPS on observation benchmark |
| W&B | Local-first (`offline` default); `--no-wandb` always available |
| TensorBoard | No |
| Eval visuals | OpenCV primary |
| Multi-agent logging | `stats_log_agents` filters which slots get flushed |
| Profiling | Python benchmarks for now; C++ profilers optional later |
| VecEnv | Not used today; when added, one `diep_sim*` per sub-env (see Risks) |
| Reward aggregates | Python callback only; C++ exports raw counters |
| Callback I/O | Episode-boundary batch log; no per-step W&B |
| Video encode | FFmpeg stdin pipe; drop frames if queue backs up |
| EpisodeStats alloc | Pre-allocate per sim; zero on `diep_reset` |

---

## Out of scope (for now)

- Video or rendering during training hot loop
- Per-step full JSON world snapshots (`include_snapshot_info=False` stays default)
- Dense reward shaping inside C++
- Parquet / DuckDB / automated replay bundles
- Offscreen OpenGL / EGL render pipeline in C++

---

## Deferred reference

Items above marked **Deferred** are documented here so they are not lost, but they are **not** part of the execution path for logged stats + eval video. Revisit when MVP is shipped and a concrete need appears (e.g. bulk run diff without video, or env FPS still too low after zero-copy).
