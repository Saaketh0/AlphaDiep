"""Verify per-step reward components are computed and normalized exactly once."""

from __future__ import annotations

import pytest

pytest.importorskip("rl.env")

from rl.env import DiepCustomParallelEnv


def _patched_env(monkeypatch, *, max_ticks=4, **kwargs):
    env = DiepCustomParallelEnv(
        seed=123,
        agents=2,
        max_ticks=max_ticks,
        scenario='rl-grid-smoke',
        observation_mode='combat',
        fast_reward_state=True,
        include_snapshot_info=False,
        **kwargs,
    )

    state_calls = {"count": 0}
    real_state_components = env._state_reward_components

    def counting_state_components(*args, **kwds):
        state_calls["count"] += 1
        return real_state_components(*args, **kwds)

    monkeypatch.setattr(env, "_state_reward_components", counting_state_components)
    return env, state_calls


def test_components_computed_once_per_step(monkeypatch):
    env, state_calls = _patched_env(
        monkeypatch,
        reward_config={'alive': 0.25, 'truncation': -0.5, 'step': -0.01},
    )
    try:
        env.reset(seed=123)
        state_calls["count"] = 0
        env.step({})
    finally:
        env.close()

    assert state_calls["count"] == 1


def test_normalizer_update_runs_once_per_step(monkeypatch):
    env, _ = _patched_env(
        monkeypatch,
        reward_config={'alive': 0.25, 'truncation': -0.5, 'step': -0.01},
        normalize_reward_components=True,
    )
    update_calls = {"true": 0, "false": 0}
    real_normalize = env.reward_normalizer.normalize_components

    def counting_normalize(components, *, update):
        update_calls["true" if update else "false"] += 1
        return real_normalize(components, update=update)

    monkeypatch.setattr(env.reward_normalizer, "normalize_components", counting_normalize)
    try:
        env.reset(seed=123)
        update_calls["true"] = 0
        update_calls["false"] = 0
        _obs, rewards, _terms, _truncs, infos = env.step({})
    finally:
        env.close()

    assert update_calls["true"] == 1
    assert update_calls["false"] == 0
    assert "reward_components_normalized" in infos["agent_0"]
    assert "reward_normalizer_state" in infos["agent_0"]
    # Normalizer state is built once and shared by reference across all agent infos.
    assert infos["agent_0"]["reward_normalizer_state"] is infos["agent_1"]["reward_normalizer_state"]


def test_rewards_unchanged_without_normalization(monkeypatch):
    """Behavioral parity: matches the existing conformance smoke (-0.26 with truncation)."""
    env, _ = _patched_env(
        monkeypatch,
        max_ticks=1,
        reward_config={'alive': 0.25, 'truncation': -0.5, 'step': -0.01},
    )
    try:
        env.reset(seed=123)
        _obs, rewards, _terms, _truncs, infos = env.step({})
    finally:
        env.close()

    assert rewards == {'agent_0': -0.26, 'agent_1': -0.26}
    assert infos['agent_0']['reward_components']['alive'] == 1.0
    assert infos['agent_0']['snapshot'] is None


def test_edge_proximity_does_not_force_snapshot(monkeypatch):
    """fast_reward_state + edge_proximity must not call _sim.snapshot()."""
    env = DiepCustomParallelEnv(
        seed=123,
        agents=2,
        max_ticks=4,
        scenario='training-ffa-easy',
        observation_mode='combat',
        fast_reward_state=True,
        include_snapshot_info=False,
        reward_config={'edge_proximity': -0.01, 'step': -0.001},
    )
    snapshot_calls = {"count": 0}

    def _fail_snapshot(*_args, **_kwargs):
        snapshot_calls["count"] += 1
        raise AssertionError("snapshot() must not be called when bounds are static")

    monkeypatch.setattr(env._sim, "snapshot", _fail_snapshot)
    try:
        env.reset(seed=123)
        _obs, _rewards, _terms, _truncs, infos = env.step({})
        assert infos['agent_0']['reward_components']['edge_proximity'] >= 0.0
        assert env._arena_bounds == (-1600.0, 1600.0, -1600.0, 1600.0)
    finally:
        # Patch the failing snapshot before close to avoid noise; close calls _sim.close, not snapshot.
        env.close()
    assert snapshot_calls["count"] == 0


def test_agent_progressions_fetched_once_per_step(monkeypatch):
    """Pre-step auto-upgrade reuses post-step cache; only one C++ fetch per step."""
    env = DiepCustomParallelEnv(
        seed=123,
        agents=2,
        max_ticks=4,
        scenario='rl-grid-smoke',
        observation_mode='combat',
        fast_reward_state=True,
        include_snapshot_info=False,
        reward_config={'alive': 0.25, 'step': -0.01},
    )
    progression_calls = {"count": 0}
    real_fetch = env._agent_progressions_array

    def counting_fetch():
        progression_calls["count"] += 1
        return real_fetch()

    monkeypatch.setattr(env, "_agent_progressions_array", counting_fetch)
    try:
        env.reset(seed=123)
        reset_calls = progression_calls["count"]
        assert reset_calls == 1

        progression_calls["count"] = 0
        env.step({})
        assert progression_calls["count"] == 1

        progression_calls["count"] = 0
        env.step({})
        assert progression_calls["count"] == 1
    finally:
        env.close()


def test_info_reward_details_only_for_logged_agents():
    """Training default: reward breakdown and static config only on logged agents."""
    env = DiepCustomParallelEnv(
        seed=123,
        agents=4,
        max_ticks=1,
        scenario='rl-grid-smoke',
        observation_mode='combat',
        fast_reward_state=True,
        include_snapshot_info=False,
        reward_config={'alive': 0.25, 'step': -0.01},
        info_log_agents=('agent_0',),
    )
    try:
        _obs, reset_infos = env.reset(seed=123)
        assert 'reward_config' in reset_infos['agent_0']
        assert 'action_shape' in reset_infos['agent_0']
        assert 'reward_config' not in reset_infos['agent_1']

        _obs, rewards, _terms, _truncs, infos = env.step({})
        assert rewards['agent_1'] == rewards['agent_0']
        assert 'reward_components' in infos['agent_0']
        assert 'reward_components' not in infos['agent_1']
        assert 'reward_config' not in infos['agent_0']
        assert 'action_shape' not in infos['agent_0']
    finally:
        env.close()


def test_include_reward_components_in_info_false():
    env = DiepCustomParallelEnv(
        seed=123,
        agents=2,
        max_ticks=1,
        scenario='rl-grid-smoke',
        observation_mode='combat',
        fast_reward_state=True,
        include_snapshot_info=False,
        reward_config={'alive': 0.25, 'step': -0.01},
        include_reward_components_in_info=False,
        info_log_agents=('agent_0',),
    )
    try:
        env.reset(seed=123)
        _obs, _rewards, _terms, _truncs, infos = env.step({})
        assert 'reward_components' not in infos['agent_0']
        assert 'raw_reward' in infos['agent_0']
    finally:
        env.close()


def test_observations_are_not_buffer_aliased_across_steps():
    """P1: stored slice views from step N must survive step N+1 untouched.

    RLlib keeps observations by reference across rollout fragments. With
    ``zero_copy_observations=True`` the env must hand out slices of a *fresh*
    parent buffer each step so old references don't get overwritten in place.
    """
    env = DiepCustomParallelEnv(
        seed=123,
        agents=2,
        max_ticks=4,
        scenario='rl-grid-smoke',
        observation_mode='combat',
        fast_reward_state=True,
        include_snapshot_info=False,
        reward_config={'alive': 0.25, 'step': -0.01},
    )
    assert env.zero_copy_observations is True
    try:
        obs_step0, _ = env.reset(seed=123)
        snapshot_step0 = {agent: obs_step0[agent]['grid_obs'].copy() for agent in obs_step0}

        obs_step1, *_ = env.step({})
        for agent in snapshot_step0:
            # Stored step-0 view must still match what we captured before step 1 ran.
            assert (obs_step0[agent]['grid_obs'] == snapshot_step0[agent]).all()
            # And the parent buffers must be different objects per step.
            assert obs_step0[agent]['grid_obs'].base is not obs_step1[agent]['grid_obs'].base
    finally:
        env.close()


def test_zero_copy_disabled_falls_back_to_copy_semantics(monkeypatch):
    """``zero_copy_observations=False`` keeps the legacy per-agent copy path."""
    env = DiepCustomParallelEnv(
        seed=123,
        agents=2,
        max_ticks=4,
        scenario='rl-grid-smoke',
        observation_mode='combat',
        fast_reward_state=True,
        include_snapshot_info=False,
        reward_config={'alive': 0.25, 'step': -0.01},
        zero_copy_observations=False,
    )
    try:
        obs, _ = env.reset(seed=123)
        # Legacy mode returns standalone arrays (no .base) because ``.copy()`` materialized them.
        assert obs['agent_0']['grid_obs'].base is None
    finally:
        env.close()


def test_reward_state_double_buffer_swaps_without_copy(monkeypatch):
    """Previous/current reward state should alternate C++ out= buffers, not allocate copies."""
    env, _ = _patched_env(
        monkeypatch,
        reward_config={'alive': 0.25, 'truncation': -0.5, 'step': -0.01},
    )
    try:
        env.reset(seed=123)
        reset_stats_buf = env._episode_stats_bufs[env._episode_stats_read_idx]

        _obs, rewards, _terms, _truncs, infos = env.step({})
        step1_stats_buf = env._episode_stats_bufs[env._episode_stats_read_idx]
        assert step1_stats_buf is not reset_stats_buf

        _obs, rewards2, _terms, _truncs, infos2 = env.step({})
        step2_stats_buf = env._episode_stats_bufs[env._episode_stats_read_idx]
        assert step2_stats_buf is reset_stats_buf
        assert step2_stats_buf is not step1_stats_buf
    finally:
        env.close()

    assert rewards['agent_0'] == rewards['agent_1']
    assert rewards2['agent_0'] == rewards2['agent_1']
    assert 'reward_components' in infos['agent_0']
    assert 'reward_components' in infos2['agent_0']
