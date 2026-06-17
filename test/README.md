# test

This folder contains TypeScript server tests. Unit tests cover low-level helpers such as coders and utilities, while e2e helpers start the built server and exercise public behavior through HTTP/WebSocket flows.

Run these through the TypeScript package:

```bash
cd ts-server
npm test
```

The helper files are shared by multiple test suites and should track `ts-server/` paths.
