const { execFileSync } = require('node:child_process');
const test = require('node:test');
const path = require('node:path');
const fs = require('node:fs');

const root = path.join(__dirname, '../..');
const dylib = path.join(root, 'cpp-build/cpp/libdiepcustom_headless_c.dylib');
const so = path.join(root, 'cpp-build/cpp/libdiepcustom_headless_c.so');

if (!fs.existsSync(dylib) && !fs.existsSync(so)) {
  execFileSync('npm', ['run', 'test:cpp'], { cwd: root, stdio: 'inherit' });
}

test('PettingZoo-compatible Python ParallelEnv wrapper exposes multi-agent actions without reward shaping', () => {
  execFileSync('uv', ['run', 'python', 'conformance/headless/python_pettingzoo_smoke.py'], { cwd: root, stdio: 'inherit' });
});

test('Combat observation builder and env smoke checks pass', () => {
  execFileSync('uv', ['run', 'python', 'conformance/headless/python_combat_observation_smoke.py'], { cwd: root, stdio: 'inherit' });
});

test('Combat observation C++ parity checks pass', () => {
  execFileSync('uv', ['run', 'python', 'conformance/headless/python_combat_observation_parity.py'], { cwd: root, stdio: 'inherit' });
});

test('PettingZoo official Parallel API test passes when optional dependencies are installed', () => {
  execFileSync('uv', ['run', 'python', 'conformance/headless/python_pettingzoo_api_test.py'], { cwd: root, stdio: 'inherit' });
});


test('Python tickless training benchmark runs batched C ABI paths', () => {
  execFileSync('uv', ['run', 'python', 'conformance/headless/python_training_benchmark.py'], { cwd: root, stdio: 'inherit' });
});

test('Combat env smoke passes', () => {
  execFileSync('uv', ['run', 'python', 'conformance/headless/python_gym_combat_wrapper_smoke.py'], { cwd: root, stdio: 'inherit' });
});
