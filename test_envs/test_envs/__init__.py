import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='MovementBandits-v0',
    entry_point='test_envs.envs:MovementBandits',
    max_episode_steps=50,
)

register(
    id='KeyDoor-v0',
    entry_point='test_envs.envs:KeyDoor',
    max_episode_steps=100,
)

register(
    id='Allwalk-v0',
    entry_point='test_envs.envs:Allwalk',
    max_episode_steps=50,
)

register(
    id='Fourrooms-v0',
    entry_point='test_envs.envs:Fourrooms',
    max_episode_steps=100,
    reward_threshold = 1,
)
