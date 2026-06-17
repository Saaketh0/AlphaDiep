from __future__ import annotations

from rl.observability.core.stats_bridge import EpisodeStatsSummary
from rl.env.headless import ABI_VERSION, EPISODE_STATS_FIELDS, DiepAction, HeadlessSim, abi_version, episode_stats_shape


def test_episode_stats_shape_matches_abi():
    assert ABI_VERSION == 12
    assert abi_version() == 12
    shape = episode_stats_shape()
    assert shape['fields'] == len(EPISODE_STATS_FIELDS) == 16
    assert shape['field_names'] == EPISODE_STATS_FIELDS


def test_episode_stats_reset_and_shots_fired():
    with HeadlessSim(seed=123, agents=1, max_ticks=8, scenario='rl-grid-smoke') as sim:
        initial = sim.episode_stats_array()
        assert initial.shape == (1, len(EPISODE_STATS_FIELDS))
        shots_fired_index = EPISODE_STATS_FIELDS.index('shots_fired')
        assert float(initial[0, shots_fired_index]) == 0.0
        sim.step([DiepAction(0, 1.0, 0.0, 1.0, 0.0, 1, 0, -1, -1)])
        after_fire = sim.episode_stats_array()
        assert float(after_fire[0, 0]) == 1.0
        assert float(after_fire[0, shots_fired_index]) == 1.0
        sim.reset(123)
        reset = sim.episode_stats_array()
        assert float(reset[0, 0]) == 0.0
        assert float(reset[0, shots_fired_index]) == 0.0


def test_episode_stats_damage_death_and_sim_isolation():
    with HeadlessSim(seed=1, agents=4, max_ticks=200, scenario='dense-collision') as dense_a, \
         HeadlessSim(seed=1, agents=4, max_ticks=200, scenario='dense-collision') as dense_b:
        idle = [DiepAction(agent_id, 0.0, 0.0, 1.0, 0.0, 0, 0, -1, -1) for agent_id in dense_a.agent_ids()]
        dense_a.step_many(idle, 10)
        stats_a = dense_a.episode_stats_array()
        stats_b = dense_b.episode_stats_array()
        enemy_kills_index = EPISODE_STATS_FIELDS.index('enemy_kills')
        damage_dealt_index = EPISODE_STATS_FIELDS.index('damage_dealt')
        enemy_damage_dealt_index = EPISODE_STATS_FIELDS.index('enemy_damage_dealt')
        assert float(stats_a[:, enemy_kills_index].sum()) > 0.0
        assert float(stats_a[:, damage_dealt_index].sum()) > 0.0
        assert float(stats_a[:, enemy_damage_dealt_index].sum()) > 0.0
        assert float(stats_b[:, enemy_kills_index].sum()) == 0.0
        assert float(stats_b[:, damage_dealt_index].sum()) == 0.0
        assert float(stats_b[:, enemy_damage_dealt_index].sum()) == 0.0


def test_episode_stats_summary_matches_current_fields():
    row = [0.0] * len(EPISODE_STATS_FIELDS)
    values = dict(zip(EPISODE_STATS_FIELDS, row))
    values.update(
        {
            'lifetime_steps': 12.0,
            'score_total': 100.0,
            'score_from_farming': 30.0,
            'score_from_pvp': 70.0,
            'damage_dealt': 11.0,
            'enemy_damage_dealt': 8.0,
            'damage_taken': 4.0,
            'shots_fired': 10.0,
            'shots_hit': 3.0,
            'enemy_kills': 2.0,
            'farm_kills': 5.0,
            'level_reached': 9.0,
        }
    )
    summary = EpisodeStatsSummary.from_row(
        [values[field] for field in EPISODE_STATS_FIELDS],
        episode_id='episode',
        controlled_agent='agent_0',
        episode_length=12,
        total_reward=1.5,
    )
    assert summary.enemy_damage_dealt == 8.0
    assert summary.enemy_kills == 2
    assert summary.farm_kills == 5
    assert summary.hit_rate == 0.3
