import gym
import numpy as np


class InnerWrapper(gym.Env):
    def __init__(self, env):
        self.env = env
        self.done = False
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def reset(self):
        self.done = False
        print("Env reset")
        return self.env.reset()

    def step(self, action):
        o, r, d, i = self.env.step(action)
        self.done = d
        if d:
            print("Environment done")
        return o, r, d, i

class MLSHEnvWraper(gym.Env):
    def __init__(self, env, subpolicies, subpolicy_time_schedule, warmup, gamma):
        self.wrapped_env = InnerWrapper(env)
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

    def reset(self):
        self.done = False
        self.cur_observation = self.wrapped_env.reset()
        self.episode_reward =0
        print("reset call by master")
        return self.cur_observation

    def step(self, action):
        if self.warmup_state:
            subpolicy = self.subpolicies[action]
            reward = 0
            done = False
            steps = 0
            for i in range(self.subpolicy_time_schedule):
                subpolicy_action, _  = subpolicy.predict(self.cur_observation)
                obs, r, done, info = self.wrapped_env.step(subpolicy_action)
                reward += r
                steps += 1
                if done:
                    break
            self.cur_observation = obs
            self.warmup_cnt += 1
            if self.warmup_cnt > self.warmup_period:
                self.warmup_state = False

        else:
            subpolicy = self.subpolicies[action]
            with subpolicy.sess.as_default():
                subpolicy.learn_async(env=self.wrapped_env, total_timesteps=self.subpolicy_time_schedule)

            obs = subpolicy.master_callback["obs"]
            reward = subpolicy.master_callback["rewards"]
            done = subpolicy.master_callback["done"]
            steps = subpolicy.master_callback["episode_len"]
            self.cur_observation = obs

        self.episode_reward += reward
        print("episode_reward", self.episode_reward)
        print("done", done)
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