# C++ Headless Core and Parity Track

This directory contains the deterministic C++ core used for parity validation and RL training. The TypeScript server in `../ts-server/` remains the production-facing reference until C++ parity is broad enough to replace it.

## Migration Contract

- Preserve external behavior first; do not replace the TypeScript runtime until C++ passes the same conformance suite.
- Maintain global/full-world deterministic parity before relying on per-agent RL observation grids.
- Keep camera/update serialization minimal and compatibility-focused; rich browser snapshots are not a headless RL goal.
- Add or expand TypeScript golden fixtures before each C++ port slice, then make C++ output match the JSON parity reports.
- Do not add new third-party C++ dependencies for the current skeleton.

## Source Layout

| Path | Purpose |
| --- | --- |
| `include/diepcustom/protocol.hpp` + `src/protocol.cpp` | Packet reader/writer compatibility |
| `include/diepcustom/physics.hpp` + `src/physics.cpp` | Deterministic vector/hash-grid/physics primitives |
| `include/diepcustom/entity_core.hpp` + `src/entity_core.cpp` | Entity manager, IDs, field groups, full-world state |
| `include/diepcustom/gameplay.hpp` + `src/gameplay.cpp` | Deterministic headless gameplay slices |
| `include/diepcustom/headless.hpp` + `src/headless.cpp` | Reusable headless simulator for RL-oriented stepping |
| `include/diepcustom/headless_c_api.h` + `src/headless_c_api.cpp` | C ABI consumed by `rl/env/headless.py` |
| `tests/` | C++ smoke/report tests |
| `tools/headless_main.cpp` | `headless_sim` CLI entry point |

Build outputs are generated under `../cpp-build/cpp/` and should stay out of source control.

## Commands

Run from `../ts-server/`:

```bash
npm run test:cpp           # CMake configure/build + C++ smoke tests
npm run test:headless      # C++ + headless Python/JS smoke checks
npm run test:parity        # C++ reports compared with TS references
npm run bench:gameplay     # TS vs C++ gameplay report runtime signal
npm run bench:headless     # in-engine headless tick throughput
npm run test:all           # broad TS/conformance/audit baseline
```

## Conformance Harness

The `../conformance/` tree contains TypeScript reference reports, golden fixtures, and parity comparators for:

- `protocol`: packet reader/writer compatibility fixtures.
- `physics`: deterministic primitive behavior.
- `entity-core`: manager/entity/full-world state snapshots and minimal compatibility packet/camera serialization.
- `gameplay`: deterministic headless gameplay slices and TS-vs-C++ report parity.

For every future C++ slice:

1. Add a deterministic TypeScript scenario to the relevant `conformance/**/report-ts.js`.
2. Regenerate/update the matching golden fixture under `conformance/fixtures/`.
3. Verify the TypeScript fixture is deterministic with the matching `golden.test.js`.
4. Implement the matching C++ report behavior.
5. Run `npm run test:cpp` and the specific comparator.
6. Run `npm run test:parity`; run `npm run bench:gameplay` when gameplay changed.

## Entity-Core Port Scope

The initial entity-core port is parity-first for headless RL training. The first RL-facing debugging artifact is the full world/entity state, not a per-agent filtered view. This avoids debugging C++ physics/entity desyncs through a quantized spatial-grid observation layer.

Acceptance priority:

1. TypeScript full-world/entity snapshot is golden.
2. C++ full-world/entity snapshot matches TypeScript exactly for deterministic fixtures.
3. Minimal camera/update packet compatibility still passes.
4. Per-agent RL observation grids are deferred until global parity is stable.

Initial TypeScript-to-C++ surface:

- `ts-server/Native/Entity.ts`: existence, state flags, IDs, hashes, deletion, wipe state.
- `ts-server/Native/Manager.ts`: ID allocation/reuse, hash table increments, slot deletion, classification, clear behavior.
- `ts-server/Native/FieldGroups.ts`: defaults, mutation state bits, no-op assignment, wipe behavior, scoreboard/camera stat arrays.
- `ts-server/Entity/Object.ts`: relations/physics/position/style groups, z-index, position/physics/style updates.
- `ts-server/Native/Camera.ts`: minimal compatibility-only defaults/state for protocol fixtures.
- `ts-server/Native/UpcreateCompiler.ts`: minimal creation/update byte snapshots for deterministic legacy protocol compatibility.

The golden report intentionally avoids full gameplay classes and client/network state. Camera/update serialization remains in conformance only as a minimal compatibility guard for the existing TypeScript/client protocol.

## Gameplay Parity Slices

Current deterministic gameplay coverage includes:

1. `overlapping-living-entities-damage`
2. `score-on-kill-and-death-removal`
3. `owner-propagated-projectile-kill-score`
4. `projectile-movement-and-lifetime`
5. `camera-player-score-integration`
6. `arena-bounds-clamp-and-can-escape`
7. `team-owner-collision-rules`
8. `collision-eligibility-filters`
9. `solid-wall-projectile-contact`

Important gotchas:

- The gameplay C++ code is still a parity report model, not a full engine/server.
- `conformance/gameplay/report-ts.js` uses a deliberately headless fake game and minimal arena object.
- Full-world snapshots remain the ground truth; local RL grids should be derived only after global parity is stable.
- Keep synthetic fixtures deterministic; avoid uncontrolled random knockback.
- `PhysicsFlags.canEscapeArena` is `1 << 8` (`256`). Do not copy older fixture flag values as `canEscapeArena`.
- Solid-wall behavior depends on owner/team checks inside `ObjectEntity.receiveKnockback`; preserve that ordering.

## Headless Simulator Layer

The C++ migration includes a reusable headless simulator layer for RL-oriented throughput, separate from WebSocket/HTTP/browser server code.

Current entry points:

- `diepcustom::headless::Simulation` in `include/diepcustom/headless.hpp`
- `../cpp-build/cpp/headless_sim` with `--seed`, `--agents`, `--ticks`, `--scenario`, and optional `--snapshot-json`
- `diepcustom_headless_c` shared library consumed by `rl/env/headless.py`

Determinism policy:

- Tick stepping is fixed and synchronous; no wall-clock, socket, or browser client state participates in `Simulation::step`.
- RNG is standard-library-only and seeded; snapshots include RNG state and draw count.
- Collision pairs are sorted by stable entity IDs before resolution.
- Training rewards are emitted separately from legacy score while legacy score/reward state remains present in snapshots.

Current limitations:

- The C++ headless code is not a drop-in server replacement.
- Real tank/shape/AI constructors are not fully ported into the layer yet.
- Per-env ghost diversity and long-run training reliability are handled in the Python/RL layer, not here.
