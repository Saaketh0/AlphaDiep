# conformance

This folder is the cross-runtime correctness harness. It generates TypeScript golden fixtures, runs C++ parity reports, compares outputs, and smoke-tests the Python headless/PettingZoo wrappers. Use it when changing TypeScript gameplay behavior, porting logic into C++, or validating RL environment wiring.

Most commands are exposed through `ts-server/package.json`, such as `npm run test:conformance`, `npm run test:parity`, and headless smoke tests.
