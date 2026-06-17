# cpp

This folder contains the deterministic C++ headless core: protocol, physics, entity-core, gameplay, simulator, C ABI, tools, and smoke tests. Python RL calls the shared library through `rl/env/headless.py`; TypeScript remains the reference implementation until parity is proven. Build output belongs in `../cpp-build/cpp/`, not here.

See `PARITY.md` for detailed migration and conformance notes.
