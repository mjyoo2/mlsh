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
        self.gamma = gamma
        self.warmup_period = warmup
        self.warmup_state=  False
        self.warmup_cnt = 0

        self.debug_histogram = np.zeros(shape=self.action_space.shape)


    def reset(self):
        self.done = False
        self.cur_observation = self.wrapped_env.reset()
        return self.cur_observation

    def step(self, action):
        if self.warmup_state:
            subpolicy = self.subpolicies[action]
            subpolicy_action, _  = subpolicy.predict(self.cur_observation)
            obs, reward, done, info = self.wrapped_env.step(subpolicy_action)
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
            self.cur_observation = obs
        return obs, reward, done, {}

    def traj_segement_generator(self, horizon, policy):
        step = 0
        action = self.wrapped_env.action_space.sample()  # not used, just so we have the datatype
        observation = np.copy(self.cur_observation)
        self.subpolicy_actual_step = 0
        self.subpolicy_reward_sum = 0
        gamma = 1
        cur_ep_ret = 0  # return in current episode
        current_it_len = 0  # len of current iteration
        current_ep_len = 0  # len of current episode
        cur_ep_true_ret = 0
        ep_true_rets = []
        ep_rets = []  # returns of completed episodes in this segment
        ep_lens = []  # Episode lengths

        # Initialize history arrays
        observations = np.array([observation for _ in range(horizon)])
        true_rewards = np.zeros(horizon, 'float32')
        rewards = np.zeros(horizon, 'float32')
        vpreds = np.zeros(horizon, 'float32')
        episode_starts = np.zeros(horizon, 'bool')
        dones = np.zeros(horizon, 'bool')
        actions = np.array([action for _ in range(horizon)])
        states = policy.initial_state
        episode_start = True  # marks if we're on first timestep of an episode
        done = False

        while True:
            action, vpred, states, _ = policy.step(observation.reshape(-1, *observation.shape), states, done)
            # Slight weirdness here because we need value function at time T
            # before returning segment [0, T-1] so we get the correct
            # terminal value
            if step > 0 and step % horizon == 0 and not done:
                yield {
                    "observations": observations,
                    "rewards": rewards,
                    "dones": dones,
                    "episode_starts": episode_starts,
                    "true_rewards": true_rewards,
                    "vpred": vpreds,
                    "actions": actions,
                    "nextvpred": vpred[0] * (1 - episode_start),
                    "ep_rets": ep_rets,
                    "ep_lens": ep_lens,
                    "ep_true_rets": ep_true_rets,
                    "total_timestep": current_it_len,
                    "end": done
                }
                _, vpred, _, _ = policy.step(observation.reshape(-1, *observation.shape))
                # Be careful!!! if you change the downstream algorithm to aggregate
                # several of these batches, then be sure to do a deepcopy
                ep_rets = []
                ep_true_rets = []
                ep_lens = []
                # Reset current iteration length
                current_it_len = 0
            i = step % horizon
            observations[i] = observation
            vpreds[i] = vpred[0]
            actions[i] = action[0]
            episode_starts[i] = episode_start

            clipped_action = action
            # Clip the actions to avoid out of bound error
            if isinstance(self.wrapped_env.action_space, gym.spaces.Box):
                clipped_action = np.clip(action, self.wrapped_env.action_space.low, self.wrapped_env.action_space.high)

            observation, reward, done, info = self.wrapped_env.step(clipped_action[0])
            self.subpolicy_reward_sum += reward * gamma
            gamma *= self.gamma
            self.cur_observation = observation
            self.done = done
            true_reward = reward
            rewards[i] = reward
            true_rewards[i] = true_reward
            dones[i] = done
            episode_start = done

            cur_ep_ret += reward
            cur_ep_true_ret += true_reward
            current_it_len += 1
            current_ep_len += 1
            self.subpolicy_actual_step += 1
            print(self.subpolicy_actual_step)
            if done:
                # Retrieve unnormalized reward if using Monitor wrapper
                maybe_ep_info = info.get('episode')
                if maybe_ep_info is not None:
                    cur_ep_true_ret = maybe_ep_info['r']

                ep_rets.append(cur_ep_ret)
                ep_true_rets.append(cur_ep_true_ret)
                ep_lens.append(current_ep_len)
                return

            step += 1


from sub_policy import MLSHSubpolicy
from stable_baselines.common.policies import  MlpPolicy
if __name__ == "__main__":

    env= gym.make("CartPole-v0")
    subpolicies = [MLSHSubpolicy(env=env, policy=MlpPolicy, verbose=1)]
    tempenv = MLSHEnvWraper(env=env, subpolicies=subpolicies, subpolicy_time_schedule=20, warmup=10, gamma=0.99)
    for i in range(100):
        obs = tempenv.reset()
        done = False
        while not done:
            obs, reward, done, info = tempenv.step(0)