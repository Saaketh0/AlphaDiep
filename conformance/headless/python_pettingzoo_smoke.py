from python.agents import AgentProfile, AgentRoster
from python.pettingzoo_env import DiepCustomParallelEnv, REWARD_FIELDS, RewardConfig, make_reward_config
from rl.env.auto_upgrade import preset_auto_upgrade_policy
from rl.runtime.rewards import BASIC_REWARD_CONFIG, training_env_config
from rl.env.rewards import RewardComponentNormalizer, _level_milestones_crossed, _ratio_delta, _retreat_from_values, weighted_rewards


def assert_default_builds_prioritize_combat_stats():
    for build_name in ('predator', 'pentashot', 'fighter', 'annihilator'):
        policy = preset_auto_upgrade_policy(build_name)
        progression = {
            'stat_levels': [0] * 8,
            'legal_stat_upgrades': [1] * 8,
            'legal_tank_upgrades': [0] * 6,
        }
        assert policy.stat_choice(progression) == 5
        progression['stat_levels'][5] = 7
        assert policy.stat_choice(progression) == 6
        progression['stat_levels'][6] = 7
        assert policy.stat_choice(progression) == 4


def main():
    assert_default_builds_prioritize_combat_stats()
    assert _level_milestones_crossed(1, 14) == 0.0
    assert _level_milestones_crossed(14, 15) == 1.0
    assert _level_milestones_crossed(14, 45) == 3.0
    assert _level_milestones_crossed(45, 30) == 0.0
    assert _ratio_delta(1, 3, 2, 6) == 0.5
    assert _ratio_delta(1, 3, 2, 2) == 0.0
    assert _retreat_from_values(1, 0, 0.25, 1, 0) == 0.25
    assert _retreat_from_values(-1, 0, 0.25, 1, 0) == 0.0
    normalizer = RewardComponentNormalizer(fields=('score_delta', 'enemy_damage_dealt'), clip=5.0)
    normalized = normalizer.normalize_components({
        'agent_0': {'score_delta': 10.0, 'enemy_damage_dealt': 2.0, 'enemy_kills': 1.0},
        'agent_1': {'score_delta': 0.0, 'enemy_damage_dealt': 4.0, 'enemy_kills': 2.0},
    })
    assert normalized['agent_0']['score_delta'] == 1.0
    assert 0.99 <= normalized['agent_0']['enemy_damage_dealt'] <= 1.0
    assert normalized['agent_0']['enemy_kills'] == 1.0
    assert normalized['agent_1']['enemy_kills'] == 2.0
    assert training_env_config()['normalize_reward_components'] is True
    tunable_fields = (
        'enemy_kills',
        'farm_kills',
        'level_delta',
        'level_milestone',
        'edge_proximity',
        'movement_speed',
        'retreat',
        'aim_accuracy',
        'enemy_damage_dealt',
    )
    for field in tunable_fields:
        assert field in REWARD_FIELDS
        assert getattr(make_reward_config(), field) == 0.0
    custom_config = make_reward_config(
        enemy_kills=2,
        farm_kills=3,
        level_delta=4,
        level_milestone=5,
        edge_proximity=6,
        movement_speed=7,
        retreat=8,
        aim_accuracy=9,
        enemy_damage_dealt=10,
    )
    assert custom_config.enemy_kills == 2.0
    assert custom_config.farm_kills == 3.0
    assert custom_config.level_milestone == 5.0
    assert weighted_rewards(custom_config, {'agent_0': {field: 1.0 for field in REWARD_FIELDS}})['agent_0'] == 54.0
    expected_basic_weights = {
        'score_delta': 1.0,
        'raw': 0.0,
        'step': -0.001,
        'death': -1.0,
        'health_delta': 0.0,
        'damage_taken': -0.01,
        'enemy_kills': 2.0,
        'farm_kills': 0.05,
        'level_delta': 0.02,
        'level_milestone': 0.5,
        'edge_proximity': -0.01,
        'movement_speed': 0.005,
        'retreat': 0.03,
        'aim_accuracy': 0.05,
        'enemy_damage_dealt': 0.02,
        'alive': 0.0,
        'truncation': 0.0,
    }
    for field, weight in expected_basic_weights.items():
        assert BASIC_REWARD_CONFIG[field] == weight
    try:
        make_reward_config(distance_to_center=1.0)
    except ValueError:
        pass
    else:
        raise AssertionError('unknown reward fields must be rejected')
    env = DiepCustomParallelEnv(seed=123, agents=2, max_ticks=4, scenario='rl-grid-smoke')
    try:
        observations, infos = env.reset(seed=123)
        assert env.possible_agents == ['agent_0', 'agent_1']
        assert env.agents == ['agent_0', 'agent_1']
        assert set(observations) == set(env.agents)
        assert set(observations['agent_0']) == {'grid_obs', 'self_obs', 'prev_action_obs', 'tank_type_obs'}
        assert observations['agent_0']['grid_obs'].shape == (18, 21, 21)
        assert observations['agent_0']['self_obs'].shape == (27,)
        assert observations['agent_0']['prev_action_obs'].shape == (5,)
        assert int(observations['agent_0']['tank_type_obs']) == 0
        assert infos['agent_0']['agent_id'] == 0
        first_snapshot = env.snapshot()
        observations, rewards, terminations, truncations, infos = env.step({
            'agent_0': {'move': [1.0, 0.0], 'aim': [1.0, 0.0], 'buttons': [1, 0]},
            # Missing agent_1 is an explicit no-op, not an AI fallback.
        })
        assert rewards == {'agent_0': 0.0, 'agent_1': 0.0}
        assert set(terminations) == {'agent_0', 'agent_1'}
        assert set(truncations) == {'agent_0', 'agent_1'}
        assert infos['agent_0']['tick'] == 1
        assert infos['agent_0']['raw_reward'] == 0.0
        assert first_snapshot['tick'] == 0
        assert env.snapshot()['tick'] == 1
        shaped = DiepCustomParallelEnv(seed=123, agents=2, max_ticks=4, scenario='rl-grid-smoke', reward_fn=lambda _env, _result, _snapshot: {'agent_1': 2.5})
        try:
            shaped.reset(seed=123)
            _obs, shaped_rewards, _terms, _truncs, _infos = shaped.step({})
            assert shaped_rewards == {'agent_0': 0.0, 'agent_1': 2.5}
        finally:
            shaped.close()
        configured = DiepCustomParallelEnv(
            seed=123,
            agents=2,
            max_ticks=1,
            scenario='rl-grid-smoke',
            reward_config={'alive': 0.25, 'death': -1.0, 'truncation': -0.5, 'step': -0.01},
        )
        try:
            _reset_obs, reset_infos = configured.reset(seed=123)
            assert configured.set_reward_config(alive=0.25, death=-1.0, truncation=-0.5, step=-0.01) == make_reward_config(alive=0.25, death=-1.0, truncation=-0.5, step=-0.01)
            assert reset_infos['agent_0']['reward_config'] == make_reward_config(alive=0.25, death=-1.0, truncation=-0.5, step=-0.01)
            _obs, configured_rewards, _terms, truncs, infos = configured.step({})
            assert configured_rewards == {'agent_0': -0.26, 'agent_1': -0.26}
            assert infos['agent_0']['reward_components']['alive'] == 1.0
            assert infos['agent_0']['reward_components']['truncation'] == 1.0
            assert 'reward_config' not in infos['agent_0']
            assert all(truncs.values())
        finally:
            configured.close()
        fast = DiepCustomParallelEnv(
            seed=123,
            agents=2,
            max_ticks=1,
            scenario='rl-grid-smoke',
            fast_reward_state=True,
            include_snapshot_info=False,
            reward_config={'alive': 0.25, 'truncation': -0.5, 'step': -0.01},
        )
        try:
            observations, _infos = fast.reset(seed=123)
            assert set(observations['agent_0']) == {'grid_obs', 'self_obs', 'prev_action_obs', 'tank_type_obs'}
            _obs, fast_rewards, _terms, _truncs, fast_infos = fast.step({})
            assert fast_rewards == {'agent_0': -0.26, 'agent_1': -0.26}
            assert fast_infos['agent_0']['snapshot'] is None
            assert fast_infos['agent_0']['reward_components']['alive'] == 1.0
            for field in tunable_fields:
                assert field in fast_infos['agent_0']['reward_components']
                assert fast_infos['agent_0']['reward_components'][field] >= 0.0
            for bounded_field in ('level_milestone', 'edge_proximity', 'movement_speed', 'retreat', 'aim_accuracy'):
                assert 0.0 <= fast_infos['agent_0']['reward_components'][bounded_field] <= 1.0
        finally:
            fast.close()
        basic = DiepCustomParallelEnv(
            seed=123,
            agents=2,
            max_ticks=1,
            scenario='rl-grid-smoke',
            fast_reward_state=True,
            include_snapshot_info=False,
            reward_config=BASIC_REWARD_CONFIG,
        )
        try:
            basic.reset(seed=123)
            _obs, basic_rewards, _terms, _truncs, basic_infos = basic.step({})
            expected_basic_rewards = weighted_rewards(BASIC_REWARD_CONFIG, {
                agent: basic_infos[agent]['reward_components'] for agent in basic_rewards
            })
            assert basic_rewards == expected_basic_rewards
            assert BASIC_REWARD_CONFIG['score_delta'] == 1.0
            assert BASIC_REWARD_CONFIG['raw'] == 0.0
            for field in tunable_fields:
                assert field in basic_infos['agent_0']['reward_components']
        finally:
            basic.close()
        normalized_basic = DiepCustomParallelEnv(
            seed=123,
            agents=2,
            max_ticks=1,
            scenario='rl-grid-smoke',
            fast_reward_state=True,
            include_snapshot_info=False,
            reward_config=BASIC_REWARD_CONFIG,
            normalize_reward_components=True,
        )
        try:
            normalized_basic.reset(seed=123)
            _obs, normalized_rewards, _terms, _truncs, normalized_infos = normalized_basic.step({})
            expected_normalized_rewards = weighted_rewards(BASIC_REWARD_CONFIG, {
                agent: normalized_infos[agent]['reward_components_normalized'] for agent in normalized_rewards
            })
            assert normalized_rewards == expected_normalized_rewards
            assert normalized_infos['agent_0']['reward_components']['enemy_kills'] == normalized_infos['agent_0']['reward_components_normalized']['enemy_kills']
            assert 'reward_normalizer_state' in normalized_infos['agent_0']
        finally:
            normalized_basic.close()
        upgrade_ready = DiepCustomParallelEnv(
            seed=123,
            agents=1,
            max_ticks=4,
            scenario='upgrade-ready',
            include_snapshot_info=False,
        )
        try:
            observations, _infos = upgrade_ready.reset(seed=123)
            self_obs = observations['agent_0']['self_obs']
            assert self_obs.shape == (27,)
            assert int(observations['agent_0']['tank_type_obs']) == 0
            assert self_obs[1] == 1.0
            assert self_obs[2] == 1.0
            assert self_obs[3] == 1.0
            step_observations, _rewards, _terms, _truncs, _infos = upgrade_ready.step({'agent_0': {'stat_upgrade_choice': 0, 'tank_upgrade_choice': 0}})
            stepped = step_observations['agent_0']
            assert stepped['prev_action_obs'].shape == (5,)
        finally:
            upgrade_ready.close()
        rostered = DiepCustomParallelEnv(
            seed=123,
            agents=4,
            max_ticks=4,
            scenario='upgrade-ready',
            include_snapshot_info=False,
            combat_builds=('predator', 'pentashot', 'fighter', 'annihilator'),
        )
        try:
            observations, _infos = rostered.reset(seed=123)
            roster = AgentRoster([
                AgentProfile(key='predator', controller=lambda _agent, _obs: {'buttons': [1, 0]}),
                AgentProfile(key='pentashot', controller=lambda _agent, _obs: {'buttons': [1, 0]}),
                AgentProfile(key='fighter', controller=lambda _agent, _obs: {'buttons': [1, 0]}),
                AgentProfile(key='annihilator', controller=lambda _agent, _obs: {'buttons': [1, 0]}),
            ])
            roster.bind(rostered.possible_agents)
            actions = roster.actions_for(observations, rostered.agents)
            assert actions['agent_0']['buttons'] == [1, 0]
            assert actions['agent_1']['buttons'] == [1, 0]
            assert actions['agent_2']['buttons'] == [1, 0]
            assert actions['agent_3']['buttons'] == [1, 0]
            stepped, _rewards, _terms, _truncs, _infos = rostered.step(actions)
            assert stepped['agent_0']['prev_action_obs'].shape == (5,)
            assert stepped['agent_1']['prev_action_obs'].shape == (5,)
            assert stepped['agent_2']['prev_action_obs'].shape == (5,)
            assert stepped['agent_3']['prev_action_obs'].shape == (5,)
        finally:
            rostered.close()
        terminated = DiepCustomParallelEnv(
            seed=123,
            agents=4,
            max_ticks=200,
            scenario='dense-collision',
            include_snapshot_info=False,
        )
        try:
            terminated.reset(seed=123)
            for _ in range(6):
                observations, _rewards, terminations, truncations, _infos = terminated.step({})
            assert all(terminations.values())
            assert all(truncations.values())
            for agent in ('agent_0', 'agent_1', 'agent_2', 'agent_3'):
                assert set(observations[agent]) == {'grid_obs', 'self_obs', 'prev_action_obs', 'tank_type_obs'}
                assert observations[agent]['grid_obs'].shape == (18, 21, 21)
                assert observations[agent]['self_obs'].shape == (27,)
                assert observations[agent]['prev_action_obs'].shape == (5)
        finally:
            terminated.close()
        assert make_reward_config(RewardConfig(raw=1.0), step=-0.1) == RewardConfig(raw=1.0, step=-0.1)
    finally:
        env.close()


if __name__ == '__main__':
    main()
