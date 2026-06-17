from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl.env import DiepCustomParallelEnv


def main() -> None:
    env = DiepCustomParallelEnv(
        seed=31,
        agents=1,
        max_ticks=4,
        scenario='upgrade-ready',
        include_snapshot_info=False,
        fast_reward_state=True,
    )
    try:
        observations, infos = env.reset(seed=31)
        assert 'agent_0' in observations
        assert env.observation_space('agent_0').contains(observations['agent_0'])
        assert isinstance(infos['agent_0'], dict)

        next_observations, rewards, terminations, truncations, step_infos = env.step({
            'agent_0': {
                'move': [0.25, -0.25],
                'aim': [0.0, 1.0],
                'buttons': [1, 0],
                'stat_upgrade_choice': -1,
                'tank_upgrade_choice': -1,
            },
        })
        assert env.observation_space('agent_0').contains(next_observations['agent_0'])
        assert isinstance(rewards['agent_0'], float)
        assert isinstance(terminations['agent_0'], bool)
        assert isinstance(truncations['agent_0'], bool)
        assert isinstance(step_infos['agent_0'], dict)
    finally:
        env.close()
    print('combat env smoke passed')


if __name__ == '__main__':
    main()
