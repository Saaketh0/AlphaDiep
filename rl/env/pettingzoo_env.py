"""Combat-only PettingZoo ParallelEnv wrapper for DiepCustom headless training."""

import os
from collections import OrderedDict

try:
    from pettingzoo import ParallelEnv
except ImportError:  # optional dependency; class still follows ParallelEnv contract
    class ParallelEnv:  # type: ignore
        metadata = {}

from .actions import action_to_diep
from .auto_upgrade import preset_auto_upgrade_policy
from .headless import AGENT_STATE_FIELDS, EPISODE_STATS_FIELDS, HeadlessSim, action_shape
from .observations import make_combat_observation_space
from .observations.combat import COMBAT_UNKNOWN_TANK_TYPE
from .rewards import (
    REWARD_FIELDS,
    RewardConfig,
    RewardComponentNormalizer,
    _level_milestones_crossed,
    _ratio_delta,
    _retreat_from_values,
    configured_rewards,
    make_reward_config,
    reward_components,
    weighted_rewards,
)
from .spaces import (
    make_action_space,
    np,
)

_AGENT_STATE_INDEX = {name: index for index, name in enumerate(AGENT_STATE_FIELDS)}
_HEALTH_INDEX = _AGENT_STATE_INDEX['health']
_SCORE_INDEX = _AGENT_STATE_INDEX['score']
_ALIVE_INDEX = _AGENT_STATE_INDEX['alive']
_X_INDEX = _AGENT_STATE_INDEX['x']
_Y_INDEX = _AGENT_STATE_INDEX['y']
_VX_INDEX = _AGENT_STATE_INDEX['vx']
_VY_INDEX = _AGENT_STATE_INDEX['vy']
_EPISODE_STATS_INDEX = {name: index for index, name in enumerate(EPISODE_STATS_FIELDS)}
_EPISODE_DAMAGE_DEALT_INDEX = _EPISODE_STATS_INDEX['damage_dealt']
_EPISODE_ENEMY_DAMAGE_DEALT_INDEX = _EPISODE_STATS_INDEX['enemy_damage_dealt']
_EPISODE_SHOTS_FIRED_INDEX = _EPISODE_STATS_INDEX['shots_fired']
_EPISODE_SHOTS_HIT_INDEX = _EPISODE_STATS_INDEX['shots_hit']
_EPISODE_ENEMY_KILLS_INDEX = _EPISODE_STATS_INDEX['enemy_kills']
_EPISODE_FARM_KILLS_INDEX = _EPISODE_STATS_INDEX['farm_kills']
_EPISODE_LEVEL_REACHED_INDEX = _EPISODE_STATS_INDEX['level_reached']
_COMBAT_SELF_RECENT_DAMAGE_RATIO_INDEX = 24
_COMBAT_SELF_RECENT_DAMAGE_DIRECTION_X_INDEX = 25
_COMBAT_SELF_RECENT_DAMAGE_DIRECTION_Y_INDEX = 26
_MOVEMENT_SPEED_NORM = 100.0
_PROGRESSION_LEVEL_INDEX = 0
_PROGRESSION_CURRENT_TANK_INDEX = 1
_PROGRESSION_STATS_AVAILABLE_INDEX = 2
_PROGRESSION_CAN_STAT_UPGRADE_INDEX = 3
_PROGRESSION_CAN_TANK_UPGRADE_INDEX = 4
_PROGRESSION_STAT_LEVELS_START_INDEX = 5
_PROGRESSION_LEGAL_STAT_LEVELS_START_INDEX = 13
_PROGRESSION_LEGAL_TANK_LEVELS_START_INDEX = 21
_DEFAULT_COMBAT_BUILDS = ('predator', 'pentashot', 'fighter', 'annihilator')

# Static arena bounds per scenario. Matches the C++ trainingScenarioConfig in
# cpp/src/headless.cpp (arenaSize / 2) and the Arena() defaults in headless.hpp.
# Lets fast_reward_state environments avoid a full snapshot just to read bounds
# for the edge_proximity reward component.
_DEFAULT_ARENA_BOUNDS = (-1000.0, 1000.0, -1000.0, 1000.0)
_SCENARIO_ARENA_BOUNDS = {
    'training-ffa-easy': (-1600.0, 1600.0, -1600.0, 1600.0),
    'training-ffa-medium': (-2400.0, 2400.0, -2400.0, 2400.0),
    'training-ffa-hard': (-3200.0, 3200.0, -3200.0, 3200.0),
}


def _scenario_arena_bounds(scenario):
    return _SCENARIO_ARENA_BOUNDS.get(scenario, _DEFAULT_ARENA_BOUNDS)


def agent_name(agent_index):
    return f'agent_{agent_index}'


def agent_index(agent_name_value):
    return int(agent_name_value.split('_', 1)[1])


def _resolve_zero_copy_observations(explicit) -> bool:
    """Decide whether combat observations should skip per-agent ``.copy()`` (P1).

    Defaults to True; set ``DIEP_ZERO_COPY_OBS=0`` (or pass ``False`` explicitly)
    to fall back to the legacy per-agent copy behavior.
    """
    if explicit is not None:
        return bool(explicit)
    raw = os.environ.get('DIEP_ZERO_COPY_OBS')
    if raw is None:
        return True
    return raw.strip().lower() not in {'0', 'false', 'no', 'off'}


def _resolve_info_log_agents(possible_agents, info_log_agents):
    if info_log_agents is None:
        return frozenset(possible_agents)
    return frozenset(info_log_agents)


class DiepCustomParallelEnv(ParallelEnv):
    """PettingZoo ParallelEnv-compatible wrapper for combat-only Python RL training."""

    metadata = {'name': 'diepcustom_headless_v1', 'render_modes': ['snapshot'], 'is_parallelizable': True}

    def __init__(
        self,
        seed=1,
        agents=1,
        max_ticks=1000,
        scenario='rl-grid-smoke',
        reward_fn=None,
        reward_config=None,
        raw_rewards=False,
        render_mode=None,
        fast_reward_state=False,
        include_snapshot_info=True,
        normalize_reward_components=False,
        include_reward_components_in_info=True,
        info_log_agents=None,
        reward_normalizer=None,
        combat_builds=_DEFAULT_COMBAT_BUILDS,
        zero_copy_observations=None,
    ):
        if agents <= 0:
            raise ValueError('agents must be positive')

        self.seed_value = seed
        self.agent_count = agents
        self.max_ticks = max_ticks
        self.scenario = scenario
        self.reward_fn = reward_fn
        self.reward_config = make_reward_config(reward_config) if reward_config is not None else None
        self.raw_rewards = raw_rewards
        self.render_mode = render_mode
        self.fast_reward_state = bool(fast_reward_state)
        self.include_snapshot_info = include_snapshot_info
        self.include_reward_components_in_info = bool(include_reward_components_in_info)
        self.normalize_reward_components = bool(normalize_reward_components)
        self.reward_normalizer = reward_normalizer or RewardComponentNormalizer()
        self.combat_builds = tuple(combat_builds or _DEFAULT_COMBAT_BUILDS)
        self.zero_copy_observations = _resolve_zero_copy_observations(zero_copy_observations)

        self._sim = HeadlessSim(seed=seed, agents=agents, max_ticks=max_ticks, scenario=scenario)
        self.possible_agents = [agent_name(i) for i in range(agents)]
        self.agents = list(self.possible_agents)
        self._info_log_agents = _resolve_info_log_agents(self.possible_agents, info_log_agents)
        self._action_shape = action_shape()
        self._combat_upgrade_policies = {
            name: preset_auto_upgrade_policy(self.combat_builds[index % len(self.combat_builds)])
            for index, name in enumerate(self.possible_agents)
        } if self.combat_builds else {}
        observation_space = make_combat_observation_space()
        self._observation_spaces = {name: observation_space for name in self.possible_agents}
        self._action_spaces = {name: make_action_space() for name in self.possible_agents}
        self._last_snapshot = None
        # Double-buffered reward state: alternate C++ out= buffers instead of copy().
        self._agent_state_bufs = [None, None]
        self._agent_state_read_idx = 0
        self._episode_stats_bufs = [None, None]
        self._episode_stats_read_idx = 0
        # Post-step progression buffer reused as pre-step input on the next step.
        self._cached_progressions = None
        # Seed arena bounds from the scenario table so edge_proximity reward shaping
        # does not force an extra C++ snapshot() per episode under fast_reward_state.
        self._arena_bounds = _scenario_arena_bounds(self.scenario)
        self._agent_progression_buffer = None
        self._combat_grid_buffer = None
        self._combat_self_buffer = None
        self._combat_prev_action_buffer = None
        self._refresh_agent_ids()

    @property
    def unwrapped(self):
        return self

    def observation_space(self, agent):
        return self._observation_spaces[agent]

    def action_space(self, agent):
        return self._action_spaces[agent]

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed_value = seed
        self._cached_progressions = None
        self._sim.reset(self.seed_value)
        self.agents = list(self.possible_agents)
        self._refresh_agent_ids()
        # Re-seed bounds from the scenario table; the live snapshot (when taken)
        # overwrites this with exact values via _update_arena_bounds.
        self._arena_bounds = _scenario_arena_bounds(self.scenario)
        snapshot = self._sim.snapshot() if self._needs_snapshot_for_step() else None
        self._update_arena_bounds(snapshot)
        post_progressions = self._agent_progressions_array()
        observations = self._observations_for(self.agents, post_progressions)
        self._cached_progressions = post_progressions
        if self.fast_reward_state:
            self._take_agent_states()
        self._take_episode_stats()
        self._last_snapshot = snapshot
        snapshot_tick = self._last_snapshot.get('tick', 0) if self._last_snapshot else 0
        infos = {}
        for agent in self.agents:
            info = {'agent_id': self._name_to_id[agent], 'snapshot_tick': snapshot_tick}
            if agent in self._info_log_agents:
                info['action_shape'] = self._action_shape
                if self.reward_config is not None:
                    info['reward_config'] = self.reward_config
            infos[agent] = info
        return observations, infos

    def step(self, actions):
        if not self.agents:
            return {}, {}, {}, {}, {}

        step_agents = list(self.agents)
        previous_snapshot = self._last_snapshot
        pre_progressions = self._cached_progressions
        if pre_progressions is None:
            pre_progressions = self._agent_progressions_array()
        result = self._sim.step(self._action_structs(step_agents, actions, pre_progressions))
        previous_agent_states, current_agent_states = self._take_agent_states()
        previous_episode_stats, current_episode_stats = self._take_episode_stats()
        snapshot = self._sim.snapshot() if self._needs_snapshot_for_step() else None
        self._update_arena_bounds(snapshot)

        self._last_snapshot = snapshot
        live_agents = self._alive_agent_names(current_agent_states)
        live_ids = {self._name_to_id[agent] for agent in live_agents}
        post_progressions = self._agent_progressions_array()
        observations = self._observations_for(step_agents, post_progressions)
        self._cached_progressions = post_progressions
        self.agents = [] if result['done'] else live_agents

        rewards, infos = self._compute_step_reward_bundle(
            result,
            snapshot,
            previous_snapshot,
            step_agents,
            previous_agent_states,
            current_agent_states,
            previous_episode_stats,
            current_episode_stats,
        )
        terminations = {agent: self._name_to_id[agent] not in live_ids for agent in step_agents}
        truncations = {agent: bool(result['done']) for agent in step_agents}
        return observations, rewards, terminations, truncations, infos

    def render(self):
        return self.snapshot()

    def snapshot(self):
        return self._sim.snapshot()

    def close(self):
        self._sim.close()

    def set_reward_config(self, config=None, **overrides):
        self.reward_config = make_reward_config(config, **overrides) if config is not None or overrides else None
        return self.reward_config

    def reset_reward_normalization(self):
        self.reward_normalizer.reset()
        return self.reward_normalizer

    def reward_components(
        self,
        result,
        snapshot,
        previous_snapshot=None,
        agents=None,
        previous_agent_states=None,
        current_agent_states=None,
        previous_episode_stats=None,
        current_episode_stats=None,
    ):
        if previous_agent_states is not None and current_agent_states is not None:
            return self._state_reward_components(result, agents, previous_agent_states, current_agent_states, previous_episode_stats, current_episode_stats)
        return reward_components(self, result, snapshot, previous_snapshot, agents, previous_episode_stats, current_episode_stats)

    def _refresh_agent_ids(self):
        self._agent_ids = self._sim.agent_ids()
        self._name_to_id = {name: self._agent_ids[i] for i, name in enumerate(self.possible_agents)}

    def _action_structs(self, agents, actions, progressions=None):
        actions = actions or {}
        enriched_actions = self._combat_actions_for(agents, actions, progressions)
        return [action_to_diep(self._name_to_id[agent], enriched_actions.get(agent)) for agent in agents]

    def _observations_for(self, agents, progressions=None):
        # P1: with zero_copy_observations=True (default) we allocate fresh parent
        # buffers per step so RLlib's stored per-agent slice views remain valid
        # across rollout fragments. The legacy path keeps reusing one ``out=`` buffer
        # and ``.copy()``-ing per agent for callers that opt out via DIEP_ZERO_COPY_OBS=0.
        zero_copy = self.zero_copy_observations
        if zero_copy:
            grid_observations = self._sim.combat_observations_array()
            self_observations = self._sim.combat_self_observations_array()
            prev_action_observations = self._sim.combat_prev_action_observations_array()
        else:
            grid_observations = self._sim.combat_observations_array(out=self._combat_grid_buffer)
            self._combat_grid_buffer = grid_observations
            self_observations = self._sim.combat_self_observations_array(out=self._combat_self_buffer)
            prev_action_observations = self._sim.combat_prev_action_observations_array(out=self._combat_prev_action_buffer)
            self._combat_prev_action_buffer = prev_action_observations
        # ``_state_retreat`` reads ``_combat_self_buffer[idx]`` on the same step we just
        # filled, so always point it at the current step's buffer regardless of mode.
        self._combat_self_buffer = self_observations
        if progressions is None:
            progressions = self._agent_progressions_array()
        if zero_copy:
            return OrderedDict(
                (
                    agent,
                    {
                        'grid_obs': grid_observations[agent_index(agent)],
                        'self_obs': self_observations[agent_index(agent)],
                        'prev_action_obs': prev_action_observations[agent_index(agent)],
                        'tank_type_obs': self._tank_type_observation(progressions[agent_index(agent)]),
                    },
                )
                for agent in agents
            )
        return OrderedDict(
            (
                agent,
                {
                    'grid_obs': grid_observations[agent_index(agent)].copy(),
                    'self_obs': self_observations[agent_index(agent)].copy(),
                    'prev_action_obs': prev_action_observations[agent_index(agent)].copy(),
                    'tank_type_obs': self._tank_type_observation(progressions[agent_index(agent)]),
                },
            )
            for agent in agents
        )

    @staticmethod
    def _tank_type_observation(progression_row):
        tank_id = int(progression_row[_PROGRESSION_CURRENT_TANK_INDEX])
        if tank_id < 0 or tank_id >= COMBAT_UNKNOWN_TANK_TYPE:
            return COMBAT_UNKNOWN_TANK_TYPE
        return tank_id

    def _take_agent_states(self):
        """Fetch agent states into the write buffer; return (previous, current) without copying."""
        if not self.fast_reward_state:
            return None, None
        write_idx = 1 - self._agent_state_read_idx
        current = self._sim.agent_states_array(out=self._agent_state_bufs[write_idx])
        self._agent_state_bufs[write_idx] = current
        previous = self._agent_state_bufs[self._agent_state_read_idx]
        self._agent_state_read_idx = write_idx
        return previous, current

    def _take_episode_stats(self):
        """Fetch episode stats into the write buffer; return (previous, current) without copying."""
        write_idx = 1 - self._episode_stats_read_idx
        current = self._sim.episode_stats_array(out=self._episode_stats_bufs[write_idx])
        self._episode_stats_bufs[write_idx] = current
        previous = self._episode_stats_bufs[self._episode_stats_read_idx]
        self._episode_stats_read_idx = write_idx
        return previous, current

    def _agent_progressions_array(self):
        progressions = self._sim.agent_progressions_array(out=self._agent_progression_buffer)
        self._agent_progression_buffer = progressions
        return progressions

    def _progression_observation(self, progression_row):
        stat_levels = progression_row[_PROGRESSION_STAT_LEVELS_START_INDEX:_PROGRESSION_LEGAL_STAT_LEVELS_START_INDEX].copy()
        legal_stat_upgrades = progression_row[_PROGRESSION_LEGAL_STAT_LEVELS_START_INDEX:_PROGRESSION_LEGAL_TANK_LEVELS_START_INDEX].copy()
        legal_tank_upgrades = progression_row[_PROGRESSION_LEGAL_TANK_LEVELS_START_INDEX:].copy()
        if np is not None:
            stat_levels = stat_levels.astype(np.float32, copy=False)
            legal_stat_upgrades = legal_stat_upgrades.astype(np.float32, copy=False)
            legal_tank_upgrades = legal_tank_upgrades.astype(np.float32, copy=False)
        return {
            'level': float(progression_row[_PROGRESSION_LEVEL_INDEX]),
            'current_tank': float(progression_row[_PROGRESSION_CURRENT_TANK_INDEX]),
            'stats_available': float(progression_row[_PROGRESSION_STATS_AVAILABLE_INDEX]),
            'can_stat_upgrade': float(progression_row[_PROGRESSION_CAN_STAT_UPGRADE_INDEX]),
            'can_tank_upgrade': float(progression_row[_PROGRESSION_CAN_TANK_UPGRADE_INDEX]),
            'stat_levels': stat_levels,
            'legal_stat_upgrades': legal_stat_upgrades,
            'legal_tank_upgrades': legal_tank_upgrades,
        }

    def _progression_observations_for(self, agents, progressions):
        return OrderedDict(
            (agent, self._progression_observation(progressions[agent_index(agent)]))
            for agent in agents
        )

    def _combat_actions_for(self, agents, actions, progressions=None):
        if progressions is None:
            progressions = self._cached_progressions
        if progressions is None:
            progressions = self._agent_progressions_array()
        progression_obs = self._progression_observations_for(agents, progressions)
        enriched = OrderedDict()
        for agent in agents:
            source_action = actions.get(agent)
            action_value = self._combat_action_with_auto_upgrade(agent, source_action, progression_obs[agent])
            enriched[agent] = action_value
        return enriched

    def _combat_action_with_auto_upgrade(self, agent, action, progression):
        policy = self._combat_upgrade_policies.get(agent)
        if policy is None:
            return action
        if isinstance(action, dict):
            return policy.apply(action, progression)
        defaults = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0, -1.0]
        try:
            raw_values = list(action)
        except TypeError:
            raw_values = []
        values = list(defaults)
        for index, value in enumerate(raw_values[:8]):
            values[index] = value
        upgrade_action = policy.apply({}, progression)
        values[6] = float(upgrade_action['stat_upgrade_choice'])
        values[7] = float(upgrade_action['tank_upgrade_choice'])
        return values

    def _raw_reward_map(self, result, agents=None):
        agents = self.agents if agents is None else agents
        raw = result.get('rewards', [])
        return {agent: float(raw[agent_index(agent)]) if agent_index(agent) < len(raw) else 0.0 for agent in agents}

    def _alive_agent_names(self, agent_states=None):
        if np is not None and agent_states is not None and hasattr(agent_states, 'shape'):
            return [agent for agent in self.possible_agents if bool(agent_states[agent_index(agent), _ALIVE_INDEX])]
        alive = self._sim.alive_mask()
        return [agent for agent in self.possible_agents if agent_index(agent) < len(alive) and alive[agent_index(agent)]]

    @property
    def _episode_damage_dealt_index(self):
        return _EPISODE_DAMAGE_DEALT_INDEX

    @property
    def _episode_enemy_damage_dealt_index(self):
        return _EPISODE_ENEMY_DAMAGE_DEALT_INDEX

    @property
    def _episode_shots_fired_index(self):
        return _EPISODE_SHOTS_FIRED_INDEX

    @property
    def _episode_shots_hit_index(self):
        return _EPISODE_SHOTS_HIT_INDEX

    @property
    def _episode_enemy_kills_index(self):
        return _EPISODE_ENEMY_KILLS_INDEX

    @property
    def _episode_farm_kills_index(self):
        return _EPISODE_FARM_KILLS_INDEX

    @property
    def _episode_level_reached_index(self):
        return _EPISODE_LEVEL_REACHED_INDEX

    def _update_arena_bounds(self, snapshot):
        if not snapshot:
            return
        arena = snapshot.get('arena') or {}
        if not arena:
            return
        self._arena_bounds = (
            float(arena.get('leftX', 0.0)),
            float(arena.get('rightX', 0.0)),
            float(arena.get('topY', 0.0)),
            float(arena.get('bottomY', 0.0)),
        )

    def _state_edge_proximity(self, current):
        if self._arena_bounds is None:
            return 0.0
        left, right, top, bottom = self._arena_bounds
        half_extent = max(1.0, min(right - left, bottom - top) * 0.5)
        x = float(current[_X_INDEX])
        y = float(current[_Y_INDEX])
        min_distance = min(x - left, right - x, y - top, bottom - y)
        return max(0.0, min(1.0, 1.0 - (min_distance / half_extent)))

    @staticmethod
    def _state_movement_speed(current):
        vx = float(current[_VX_INDEX])
        vy = float(current[_VY_INDEX])
        return max(0.0, min(1.0, ((vx * vx + vy * vy) ** 0.5) / _MOVEMENT_SPEED_NORM))

    @staticmethod
    def _episode_stat_delta(previous_episode_stats, current_episode_stats, idx, field_index):
        if previous_episode_stats is None or current_episode_stats is None:
            return 0.0
        try:
            return float(current_episode_stats[idx, field_index] - previous_episode_stats[idx, field_index])
        except (IndexError, TypeError):
            return 0.0

    @staticmethod
    def _episode_stat_value(episode_stats, idx, field_index):
        if episode_stats is None:
            return 0.0
        try:
            return float(episode_stats[idx, field_index])
        except (IndexError, TypeError):
            return 0.0

    def _state_retreat(self, idx, current):
        if self._combat_self_buffer is None:
            return 0.0
        try:
            self_obs = self._combat_self_buffer[idx]
        except (IndexError, TypeError):
            return 0.0
        return _retreat_from_values(
            current[_VX_INDEX],
            current[_VY_INDEX],
            self_obs[_COMBAT_SELF_RECENT_DAMAGE_RATIO_INDEX],
            self_obs[_COMBAT_SELF_RECENT_DAMAGE_DIRECTION_X_INDEX],
            self_obs[_COMBAT_SELF_RECENT_DAMAGE_DIRECTION_Y_INDEX],
        )

    def _state_reward_components(self, result, agents, previous_agent_states, current_agent_states, previous_episode_stats=None, current_episode_stats=None):
        agents = self.agents if agents is None else agents
        raw_rewards = self._raw_reward_map(result, agents)
        done = bool(result.get('done', False))
        components = {}
        for agent in agents:
            idx = agent_index(agent)
            previous = previous_agent_states[idx]
            current = current_agent_states[idx]
            previous_health = float(previous[_HEALTH_INDEX])
            current_health = float(current[_HEALTH_INDEX])
            is_alive = bool(current[_ALIVE_INDEX])
            components[agent] = {
                'raw': raw_rewards.get(agent, 0.0),
                'score_delta': float(current[_SCORE_INDEX] - previous[_SCORE_INDEX]),
                'health_delta': current_health - previous_health,
                'damage_taken': max(0.0, previous_health - current_health),
                'enemy_kills': self._episode_stat_delta(previous_episode_stats, current_episode_stats, idx, _EPISODE_ENEMY_KILLS_INDEX),
                'farm_kills': self._episode_stat_delta(previous_episode_stats, current_episode_stats, idx, _EPISODE_FARM_KILLS_INDEX),
                'level_delta': max(0.0, self._episode_stat_delta(previous_episode_stats, current_episode_stats, idx, _EPISODE_LEVEL_REACHED_INDEX)),
                'level_milestone': _level_milestones_crossed(
                    self._episode_stat_value(previous_episode_stats, idx, _EPISODE_LEVEL_REACHED_INDEX),
                    self._episode_stat_value(current_episode_stats, idx, _EPISODE_LEVEL_REACHED_INDEX),
                ),
                'edge_proximity': self._state_edge_proximity(current),
                'movement_speed': self._state_movement_speed(current),
                'retreat': self._state_retreat(idx, current),
                'aim_accuracy': _ratio_delta(
                    self._episode_stat_value(previous_episode_stats, idx, _EPISODE_SHOTS_HIT_INDEX),
                    self._episode_stat_value(current_episode_stats, idx, _EPISODE_SHOTS_HIT_INDEX),
                    self._episode_stat_value(previous_episode_stats, idx, _EPISODE_SHOTS_FIRED_INDEX),
                    self._episode_stat_value(current_episode_stats, idx, _EPISODE_SHOTS_FIRED_INDEX),
                ),
                'enemy_damage_dealt': self._episode_stat_delta(previous_episode_stats, current_episode_stats, idx, _EPISODE_ENEMY_DAMAGE_DEALT_INDEX),
                'alive': 1.0 if is_alive else 0.0,
                'death': 0.0 if is_alive else 1.0,
                'truncation': 1.0 if done else 0.0,
                'step': 1.0,
            }
        return components

    def _needs_snapshot_for_step(self):
        # edge_proximity no longer forces a snapshot: _arena_bounds is seeded
        # from the static scenario table in __init__/reset.
        return (
            self.include_snapshot_info
            or self.reward_fn is not None
            or (self.reward_config is not None and not self.fast_reward_state)
        )

    def _compute_step_reward_bundle(
        self,
        result,
        snapshot,
        previous_snapshot,
        agents,
        previous_agent_states,
        current_agent_states,
        previous_episode_stats,
        current_episode_stats,
    ):
        """Compute rewards and infos in a single pass.

        Replaces the old ``_rewards`` + ``_infos`` split that ran the reward
        component pipeline twice per step (once for weighted rewards, once for
        infos) and re-normalized 20x. Components are computed once; the
        normalizer is updated once; ``reward_normalizer.state()`` is built once
        and shared by reference across all agent info dicts.
        """
        agents = self.agents if agents is None else agents
        raw_rewards = self._raw_reward_map(result, agents)
        tick = int(result.get('tick', snapshot.get('tick', 0) if snapshot else 0))

        components: dict | None = None
        normalized_components: dict | None = None

        # Compute components once; both rewards and infos consume the same dict.
        components = self.reward_components(
            result,
            snapshot,
            previous_snapshot,
            agents,
            previous_agent_states,
            current_agent_states,
            previous_episode_stats,
            current_episode_stats,
        )

        if self.reward_fn is not None:
            produced = self.reward_fn(self, result, snapshot)
            rewards = {agent: float(produced.get(agent, 0.0)) for agent in agents}
        elif self.reward_config is not None:
            if self.normalize_reward_components:
                # One update per step; normalized view is reused by both rewards and infos.
                normalized_components = self.reward_normalizer.normalize_components(components, update=True)
                rewards = weighted_rewards(self.reward_config, normalized_components)
            else:
                rewards = weighted_rewards(self.reward_config, components)
        elif self.raw_rewards:
            rewards = dict(raw_rewards)
        else:
            rewards = {agent: 0.0 for agent in agents}

        # Build normalizer state once; share by reference across agent infos.
        normalizer_state = self.reward_normalizer.state() if normalized_components is not None else None

        infos = {}
        for agent in agents:
            info = {
                'agent_id': self._name_to_id[agent],
                'tick': tick,
                'raw_reward': raw_rewards[agent],
                'snapshot': snapshot if self.include_snapshot_info else None,
            }
            if (
                self.include_reward_components_in_info
                and agent in self._info_log_agents
                and components is not None
            ):
                info['reward_components'] = components[agent]
                if normalized_components is not None:
                    info['reward_components_normalized'] = normalized_components[agent]
                    info['reward_normalizer_state'] = normalizer_state
            infos[agent] = info
        return rewards, infos


parallel_env = DiepCustomParallelEnv


__all__ = [
    'DiepCustomParallelEnv', 'parallel_env', 'RewardConfig', 'REWARD_FIELDS',
    'make_reward_config', 'reward_components', 'configured_rewards', 'action_to_diep',
    'agent_name', 'agent_index',
]
