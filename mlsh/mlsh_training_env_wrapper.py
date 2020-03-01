import gym
import numpy as np
from stable_baselines.ppo2.ppo2 import get_schedule_fn


class MLSHEnvWraper(gym.Env):
    """
    Environment wrapper for master policy.
    """
    def __init__(self, env, subpolicies, subpolicy_time_schedule, warmup, gamma):
        """
        :param env: environment to
        :param subpolicies: subpolicies to learn or execute
        :param subpolicy_time_schedule: subpolicie's time schedule
        :param warmup: warmup period for master policy
        :param gamma: discount factor; not used yet
        """
        self.wrapped_env = env
        self.warmup = warmup
        self.subpolicy_time_schedule = get_schedule_fn(subpolicy_time_schedule)
        self.done = False
        self.cur_observation = None
        self.subpolicy_actual_step = 0
        self.subpolicy_reward_sum = 0
        self.subpolicies = subpolicies

        self.warmup_period = warmup
        self.warmup_state=  False
        self.warmup_cnt = 0
        self.gamma = gamma
        self.debug_histogram = np.zeros(shape=self.action_space.shape)
        self.time_step = 0
        self.episode_reward = 0

    @property
    def observation_space(self):
        return self.wrapped_env.observation_space

    @property
    def action_space(self):
        return gym.spaces.Discrete(len(self.subpolicies))

    def reset(self):
        self.time_step = 0
        self.done = False
        self.cur_observation = self.wrapped_env.reset()
        self.episode_reward =0
        return self.cur_observation

    def step(self, action):
        subpolicy = self.subpolicies[action]
        reward = 0
        done = False
        steps = 0
        obs = self.cur_observation
        subpolicy_time = int(self.subpolicy_time_schedule(self.time_step))
        if self.warmup_cnt < self.warmup_period:
            for i in range(subpolicy_time):
                obs, r, done, info = subpolicy.rollout_async(obs, self.wrapped_env)
                self.cur_observation = obs
                reward += r
                steps += 1
                if done:
                    break
        else:
            for i in range(subpolicy_time):
                obs, r, done, info = subpolicy.rollout_async(obs, self.wrapped_env)
                subpolicy.learn_async()
                self.cur_observation = obs
                reward += r
                steps += 1
                if done:
                    break
        self.warmup_cnt += 1
        self.time_step += 1
        return obs, reward, done, {}

