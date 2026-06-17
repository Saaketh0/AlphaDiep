"""Python-side reward configuration for PettingZoo training."""

from dataclasses import dataclass
from math import floor, sqrt

_MOVEMENT_SPEED_NORM = 100.0


@dataclass(frozen=True)
class RewardConfig:
    """Declarative reward weights evaluated in Python from transition state."""

    raw: float = 0.0
    score_delta: float = 0.0
    health_delta: float = 0.0
    damage_taken: float = 0.0
    enemy_kills: float = 0.0
    farm_kills: float = 0.0
    level_delta: float = 0.0
    level_milestone: float = 0.0
    edge_proximity: float = 0.0
    movement_speed: float = 0.0
    retreat: float = 0.0
    aim_accuracy: float = 0.0
    enemy_damage_dealt: float = 0.0
    alive: float = 0.0
    death: float = 0.0
    truncation: float = 0.0
    step: float = 0.0


REWARD_FIELDS = tuple(RewardConfig.__dataclass_fields__.keys())
DENSE_REWARD_FIELDS = (
    'score_delta',
    'health_delta',
    'damage_taken',
    'edge_proximity',
    'movement_speed',
    'retreat',
    'aim_accuracy',
    'enemy_damage_dealt',
)


class RewardComponentNormalizer:
    """Scale dense reward components by running absolute magnitude.

    This intentionally does not subtract a mean: positive progress stays positive,
    penalties stay negative after weighting, and explicit RewardConfig weights still
    control importance. Rare event components are excluded by default.
    """

    def __init__(self, fields=DENSE_REWARD_FIELDS, decay=0.999, epsilon=1e-8, clip=5.0):
        self.fields = tuple(fields)
        self.decay = float(decay)
        self.epsilon = float(epsilon)
        self.clip = float(clip)
        self._scales = {field: 1.0 for field in self.fields}
        self._counts = {field: 0 for field in self.fields}

    def reset(self):
        self._scales = {field: 1.0 for field in self.fields}
        self._counts = {field: 0 for field in self.fields}

    def state(self):
        return {
            'fields': self.fields,
            'decay': self.decay,
            'epsilon': self.epsilon,
            'clip': self.clip,
            'scales': dict(self._scales),
            'counts': dict(self._counts),
        }

    def _observe_value(self, field, value):
        magnitude = abs(float(value))
        if self._counts[field] <= 0:
            if magnitude > self.epsilon:
                self._scales[field] = magnitude
        elif magnitude > self.epsilon:
            self._scales[field] = self.decay * self._scales[field] + (1.0 - self.decay) * magnitude
        self._counts[field] += 1

    def observe(self, components):
        for values in components.values():
            for field in self.fields:
                if field in values:
                    self._observe_value(field, values.get(field, 0.0))
        return self

    def normalized_value(self, field, value):
        if field not in self._scales:
            return float(value)
        scale = max(self.epsilon, self._scales[field])
        normalized = float(value) / scale
        return max(-self.clip, min(self.clip, normalized))

    def normalize_components(self, components, update=True):
        if update:
            self.observe(components)
        normalized = {}
        for agent, values in components.items():
            agent_values = dict(values)
            for field in self.fields:
                if field in agent_values:
                    agent_values[field] = self.normalized_value(field, agent_values[field])
            normalized[agent] = agent_values
        return normalized


def make_reward_config(config=None, **overrides):
    """Create a RewardConfig from None, RewardConfig, mapping, or keywords."""

    if config is None:
        values = {}
    elif isinstance(config, RewardConfig):
        values = {field: getattr(config, field) for field in REWARD_FIELDS}
    elif isinstance(config, dict):
        unknown = set(config) - set(REWARD_FIELDS)
        if unknown:
            raise ValueError(f'unknown reward config fields: {sorted(unknown)}')
        values = dict(config)
    else:
        raise TypeError('reward_config must be None, RewardConfig, or dict')
    unknown = set(overrides) - set(REWARD_FIELDS)
    if unknown:
        raise ValueError(f'unknown reward config fields: {sorted(unknown)}')
    values.update(overrides)
    return RewardConfig(**{field: float(values.get(field, 0.0)) for field in REWARD_FIELDS})


def _entity_by_agent_id(snapshot, agent_id):
    for entity in snapshot.get('entities', []):
        if entity.get('kind') == 'agent' and entity.get('id') == agent_id:
            return entity
    return None


def _score(entity):
    if not entity:
        return 0.0
    return float(entity.get('score', {}).get('score', 0.0))


def _health(entity):
    if not entity:
        return 0.0
    return float(entity.get('health', {}).get('health', 0.0))


def _level(entity):
    if not entity:
        return 0.0
    level = entity.get('level')
    if level is not None:
        return float(level)
    progression = entity.get('progression', {})
    return float(progression.get('level', 0.0))


def _level_milestones_crossed(previous_level, current_level):
    previous_bucket = max(0, min(3, floor(float(previous_level) / 15.0)))
    current_bucket = max(0, min(3, floor(float(current_level) / 15.0)))
    return float(max(0, current_bucket - previous_bucket))


def _ratio_delta(numerator_previous, numerator_current, denominator_previous, denominator_current):
    denominator_delta = float(denominator_current) - float(denominator_previous)
    if denominator_delta <= 0.0:
        return 0.0
    numerator_delta = max(0.0, float(numerator_current) - float(numerator_previous))
    return max(0.0, min(1.0, numerator_delta / denominator_delta))


def _retreat_from_values(vx, vy, recent_damage_ratio, damage_direction_x, damage_direction_y):
    damage_ratio = max(0.0, min(1.0, float(recent_damage_ratio)))
    if damage_ratio <= 0.0:
        return 0.0
    movement_magnitude = sqrt(float(vx) * float(vx) + float(vy) * float(vy))
    if movement_magnitude <= 0.000001:
        return 0.0
    direction_magnitude = sqrt(float(damage_direction_x) * float(damage_direction_x) + float(damage_direction_y) * float(damage_direction_y))
    if direction_magnitude <= 0.000001:
        return 0.0
    movement_x = float(vx) / movement_magnitude
    movement_y = float(vy) / movement_magnitude
    away_x = float(damage_direction_x) / direction_magnitude
    away_y = float(damage_direction_y) / direction_magnitude
    return max(0.0, min(1.0, (movement_x * away_x + movement_y * away_y) * damage_ratio))


def _retreat(entity):
    if not entity:
        return 0.0
    velocity = entity.get('velocity', {})
    damage = entity.get('damage', {})
    health = entity.get('health', {})
    max_health = max(1.0, float(health.get('maxHealth', 1.0)))
    return _retreat_from_values(
        velocity.get('x', 0.0),
        velocity.get('y', 0.0),
        float(damage.get('recentTaken', 0.0)) / max_health,
        damage.get('recentDirectionX', 0.0),
        damage.get('recentDirectionY', 0.0),
    )


def _movement_speed(entity):
    if not entity:
        return 0.0
    velocity = entity.get('velocity', {})
    vx = float(velocity.get('x', 0.0))
    vy = float(velocity.get('y', 0.0))
    return max(0.0, min(1.0, sqrt(vx * vx + vy * vy) / _MOVEMENT_SPEED_NORM))


def _edge_proximity(entity, snapshot):
    if not entity or not snapshot:
        return 0.0
    arena = snapshot.get('arena') or {}
    if not arena:
        return 0.0
    position = entity.get('position', {})
    x = float(position.get('x', 0.0))
    y = float(position.get('y', 0.0))
    left = float(arena.get('leftX', 0.0))
    right = float(arena.get('rightX', 0.0))
    top = float(arena.get('topY', 0.0))
    bottom = float(arena.get('bottomY', 0.0))
    half_extent = max(1.0, min(right - left, bottom - top) * 0.5)
    min_distance = min(x - left, right - x, y - top, bottom - y)
    return max(0.0, min(1.0, 1.0 - (min_distance / half_extent)))


def _stats_delta(previous_stats, current_stats, index, agent_index_value):
    if previous_stats is None or current_stats is None:
        return 0.0
    try:
        return float(current_stats[agent_index_value][index] - previous_stats[agent_index_value][index])
    except (IndexError, TypeError):
        return 0.0


def _env_stat_delta(env, previous_stats, current_stats, index_attr, agent_index_value):
    if not hasattr(env, index_attr):
        return 0.0
    return _stats_delta(previous_stats, current_stats, getattr(env, index_attr), agent_index_value)


def _env_stat_value(env, stats, index_attr, agent_index_value):
    if stats is None or not hasattr(env, index_attr):
        return 0.0
    try:
        return float(stats[agent_index_value][getattr(env, index_attr)])
    except (IndexError, TypeError):
        return 0.0


def snapshot_reward_components(env, result, snapshot, previous_snapshot, agents=None, previous_episode_stats=None, current_episode_stats=None):
    """Return unweighted transition components computed from JSON snapshots."""

    agents = env.agents if agents is None else agents
    raw_rewards = env._raw_reward_map(result, agents)
    done = bool(result.get('done', False))
    alive_set = set(env._alive_agent_names())
    components = {}
    for agent in agents:
        agent_index_value = int(agent.rsplit('_', 1)[1])
        agent_id = env._name_to_id[agent]
        previous_entity = _entity_by_agent_id(previous_snapshot or {}, agent_id)
        current_entity = _entity_by_agent_id(snapshot or {}, agent_id)
        previous_score = _score(previous_entity)
        current_score = _score(current_entity)
        previous_health = _health(previous_entity)
        current_health = _health(current_entity)
        previous_level = _env_stat_value(env, previous_episode_stats, '_episode_level_reached_index', agent_index_value)
        current_level = _env_stat_value(env, current_episode_stats, '_episode_level_reached_index', agent_index_value)
        if previous_level == 0.0 and current_level == 0.0:
            previous_level = _level(previous_entity)
            current_level = _level(current_entity)
        is_alive = agent in alive_set
        components[agent] = {
            'raw': raw_rewards.get(agent, 0.0),
            'score_delta': current_score - previous_score,
            'health_delta': current_health - previous_health,
            'damage_taken': max(0.0, previous_health - current_health),
            'enemy_kills': _env_stat_delta(env, previous_episode_stats, current_episode_stats, '_episode_enemy_kills_index', agent_index_value),
            'farm_kills': _env_stat_delta(env, previous_episode_stats, current_episode_stats, '_episode_farm_kills_index', agent_index_value),
            'level_delta': max(
                0.0,
                _env_stat_delta(env, previous_episode_stats, current_episode_stats, '_episode_level_reached_index', agent_index_value)
                if hasattr(env, '_episode_level_reached_index') else _level(current_entity) - _level(previous_entity),
            ),
            'level_milestone': _level_milestones_crossed(previous_level, current_level),
            'edge_proximity': _edge_proximity(current_entity, snapshot),
            'movement_speed': _movement_speed(current_entity),
            'retreat': _retreat(current_entity),
            'aim_accuracy': _ratio_delta(
                _env_stat_value(env, previous_episode_stats, '_episode_shots_hit_index', agent_index_value),
                _env_stat_value(env, current_episode_stats, '_episode_shots_hit_index', agent_index_value),
                _env_stat_value(env, previous_episode_stats, '_episode_shots_fired_index', agent_index_value),
                _env_stat_value(env, current_episode_stats, '_episode_shots_fired_index', agent_index_value),
            ),
            'enemy_damage_dealt': _env_stat_delta(env, previous_episode_stats, current_episode_stats, '_episode_enemy_damage_dealt_index', agent_index_value),
            'alive': 1.0 if is_alive else 0.0,
            'death': 0.0 if is_alive else 1.0,
            'truncation': 1.0 if done else 0.0,
            'step': 1.0,
        }
    return components


def weighted_rewards(config, components):
    """Evaluate weighted reward values from precomputed components."""

    config = make_reward_config(config)
    return {
        agent: sum(getattr(config, field) * values.get(field, 0.0) for field in REWARD_FIELDS)
        for agent, values in components.items()
    }


def configured_rewards(config, env, result, snapshot, previous_snapshot, agents=None, previous_episode_stats=None, current_episode_stats=None):
    return weighted_rewards(
        config,
        snapshot_reward_components(env, result, snapshot, previous_snapshot, agents, previous_episode_stats, current_episode_stats),
    )


# Backward-compatible public name used by existing scripts/tests.
reward_components = snapshot_reward_components


__all__ = [
    'RewardConfig', 'REWARD_FIELDS', 'DENSE_REWARD_FIELDS', 'RewardComponentNormalizer',
    'make_reward_config', 'snapshot_reward_components', 'reward_components',
    'weighted_rewards', 'configured_rewards',
]
