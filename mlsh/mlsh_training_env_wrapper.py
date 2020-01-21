import gym
import numpy as np



class MLSHEnvWraper(gym.Env):
    def __init__(self, env, subpolicies, subpolicy_time_schedule, warmup, gamma):
        self.wrapped_env = env
        self.observation_space = env.observation_space
        self.action_space = gym.spaces.Discrete(len(subpolicies))
        self.warmup = warmup
        self.subpolicy_time_schedule = subpolicy_time_schedule
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

        self.episode_reward = 0

    def reset(self):
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
        if self.warmup_cnt < self.warmup_period:
            for i in range(self.subpolicy_time_schedule):
                obs, r, done, info = subpolicy.rollout_async(obs, self.wrapped_env)
                self.cur_observation = obs
                reward += r
                steps += 1
                if done:
                    break
        else:
            for i in range(self.subpolicy_time_schedule):
                obs, r, done, info = subpolicy.rollout_async(obs, self.wrapped_env)
                subpolicy.learn_async()
                self.cur_observation = obs
                reward += r
                steps += 1
                if done:
                    break
        self.warmup_cnt += 1
        return obs, reward, done, {}


from sub_policy import MLSHSubpolicy
from stable_baselines.common.policies import  MlpPolicy

if __name__ == "__main__":

    env= gym.make("CartPole-v0")
    subpolicies = [MLSHSubpolicy(env=env, policy=MlpPolicy, verbose=0)]
    tempenv = MLSHEnvWraper(env=env, subpolicies=subpolicies, subpolicy_time_schedule=20, warmup=10, gamma=0.99)
    for i in range(100):
        obs = tempenv.reset()
        done = False
        while not done:
            obs, reward, done, info = tempenv.step(0)