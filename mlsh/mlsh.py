import time

from stable_baselines.common.policies import RecurrentActorCriticPolicy
from stable_baselines.common.base_class import BaseRLModel, _UnvecWrapper
from stable_baselines.ppo2 import PPO2
from stable_baselines.common.runners import AbstractEnvRunner
from stable_baselines.common import make_vec_env
from stable_baselines.ppo2.ppo2 import swap_and_flatten
from stable_baselines.ppo2.ppo2 import Runner as PPORunner
import numpy as np
import gym



class MLSH(BaseRLModel):
    def __init__(self, master_policy, env, subpolicy_timestep=200, num_subpolicy=3,
                 , master_lr=1e-2, subpolicy_lr=3e-4, tensorboard_log=None, gamma=0.99, lam=0.95,
                 subpolicy_model_policy=None, *args, **kwargs):
        super().__init__(policy=master_policy, env=env, verbose=kwargs.get('verbose', 0),
                         policy_base=None, requires_vec_env=True)

        assert not isinstance(master_policy, RecurrentActorCriticPolicy), "Error Recurrent Policy is not implemented yet"
        assert not isinstance(subpolicy_model_policy, RecurrentActorCriticPolicy), "Error Recurrent Policy is not" \
                                                                                    "implemented yet"
        if env is not None:
            self.observation_space = env.observation_space
            self.action_space = env.action_space
        if subpolicy_model_policy is None:
            subpolicy_model_policy = master_policy
        self.observation_space = env.observation_space
        self.num_subpolicy = num_subpolicy
        # defaults for creates master model
        self.tempenv = make_vec_env(get_temp_env(observation_space=self.observation_space, num_subpolicy=num_subpolicy),
                                    n_envs=env.n_envs)
        self.master_lr = master_lr
        self.master_policy_arch = master_policy
        self.master_model = PPO2(env=self.tempenv, policy=self.master_policy_arch, learning_rate=self.master_lr)

        self.subpolicy_model_learning_rate = subpolicy_lr
        self.tensorboard_log=tensorboard_log
        self.gamma = gamma
        self.subpolicies =[]
        self.subpolicy_runners = []
        self.subpolicy_timestep = subpolicy_timestep
        self.current_subpolicy = 0
        self.lam = lam

        for i in range(num_subpolicy):
            self.subpolicies.append(PPO2(subpolicy_model_policy, env).policy)
        for i in range(num_subpolicy):
            self.subpolicy_runners.append(PPORunner(env= self.env, model = self.subpolicies[i],
                                                    n_steps=self.subpolicy_timestep, gamma=self.gamma, lam=self.lam))


    def master_action_probability(self, observation, state=None, mask=None, actions=None, logp=False):
        return self.master_model.action_probability(observation, state, mask, actions, logp)

    def subpolicy_action_probability(self, subpolicy_index, observation, state=None, mask=None, actions=None, logp=False):
        return self.subpolicies[subpolicy_index].action_probability\
            (observation, state, mask, actions, logp)

    def action_probability(self, observation, state=None, mask=None, actions=None, logp=False):
        return self.subpolicy_action_probability(subpolicy_index=self.current_subpolicy, observation=observation,
                                                 state=state, mask=mask, actions=actions, logp=logp)

    def learn(self, total_timesteps, callback=None, log_interval=100, tb_log_name="run",
              reset_num_timesteps=True):

        # initialize theta
        self.master_model = PPO2(env=self.tempenv, policy=self.master_policy_arch, learning_rate=self.master_lr)



        return self


class MLSHRunner(AbstractEnvRunner):
    MB_OBS= 0
    MB_REWARDS = 1
    MB_ACTIONS = 2
    MB_VALUES = 3
    MB_NEGLOGPACS =4
    MB_DONES = 5

    def __init__(self, *, env, master_model, subpolicies, n_steps, gamma, lam, subpolicy_timestep):
        """
        The runner class corresponds to rollout in the original MLSH codes
        :param env:  environment to learn
        :param model: whole mlsh classes
        :param n_steps: the number of steps for sampling
        :param gamma: discount factor
        :param lam: bias-variance tradeoff for advantage
        :param subpolicy_timestep: the length where master action is valid
        """
        super().__init__(env=env, model=master_model, n_steps=n_steps)
        self.gamma = gamma
        self.lam = lam
        self.macrolen = subpolicy_timestep
        self.subpolicies = subpolicies
        self.subpolicy_timestep = subpolicy_timestep
        self.macro_n_steps = n_steps // subpolicy_timestep
        self.current_subpolicy = []
        self.master_model = self.model  # alias to avoid confusion; i.e., the master model is exactly same to model

    def run(self):
        """
         Run a learning step of the model
         :return:

             - master_observations: (np.ndarray) the observations of master_policies
             - master_rewards: the rewards of master policies
             - master_actions: (np.ndarray) the actions of master policy
             - master_values: (np.ndarray) the value function ouput from the master policy
             - master masks: (numpy bool) wheter an episodes is over or not in the master time scopes
             - master negative log probabilities: (np.ndarray): master's log probabilities
             - subpolicy_minibatch: (dicts) the mini batches containing following objects
                    key:
                        the indices of subpolicies
                    values: lists of tuples
                          - observations: (np.ndarray) the observations
                          - rewards: (np.ndarray) the rewards
                          - values: (np.ndarray) the value function output
                          - masks: (numpy bool) whether an episode is over or not
                          - actions: (np.ndarray) the actions of subpolicies
                          - negative log probabilities: (np.ndarray)
             - infos: (dict) the extra information of the model

         """
        # mb stands for minibatch

        master_obs, master_rewards, master_actions, master_values, master_dones, master_neglogpacs = [], [], [], [], [], []
        mb_states = self.states
        ep_infos = []
        subpolicy_minibatch = {}
        for i in range(len(self.subpolicies)):
            subpolicy_minibatch[i] = [ [], [], [], [], [], [] ]  # 6 lists, mb_obs, mb_returns, mb_dones, mv_values, \
                                                                 # mb_neglopacs for each.

        # Roll out master policy
        for _ in range(self.n_steps):
            actions, values, self.states, neglogpacs = self.model.step(self.obs, self.states, self.dones)
            master_obs.append(self.obs.copy())
            master_actions.append(actions)
            master_values.append(values)
            master_neglogpacs.append(neglogpacs)
            master_dones.append(self.dones)
            clipped_actions = actions

            # Choose subpolicy
            chosen_subpolicy = []
            for a in actions:
                self.current_subpolicy.append(self.subpolicies[a])

            actions, values, self.states, neglogpacs = self.master_model.step(self.obs, self.states, self.dones)
            self.current_subpolicy = actions
            # roll out subpolicy

            mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglopacs = [], [], [], [], [], []
            for __ in range(self.subpolicy_timestep):
                subpolicy_actions, subpolicy_values, states, subpolicy_neglopcacs =self.subpolicy_step(self.obs, self.states, self.dones)
                cursor = 0

                # each subpolicy has its own minibatches
                for o, a, v, n, d in zip(self.obs, subpolicy_actions, subpolicy_values, subpolicy_neglopcacs, self.dones):
                    idx = self.current_subpolicy[cursor]
                    subpolicy_minibatch[idx][MLSHRunner.MB_OBS].append(o.copy())
                    subpolicy_minibatch[idx][MLSHRunner.MB_ACTIONS].append(a)
                    subpolicy_minibatch[idx][MLSHRunner.MB_VALUES].append(v)
                    subpolicy_minibatch[idx][MLSHRunner.MB_NEGLOGPACS].append(n)
                    subpolicy_minibatch[idx][MLSHRunner.MB_DONES].append(d)
                    cursor += 1

                if isinstance(self.env.action_space, gym.spaces.Box):
                    clipped_actions = np.clip(subpolicy_actions, self.env.action_space.low, self.env.action_space.high)

                self.obs[:], rewards, self.dones, infos = self.env.step(clipped_actions)

                for info in infos:
                    maybe_ep_info = info.get('episode')
                    if maybe_ep_info is not None:
                        ep_infos.append(maybe_ep_info)
                cursor = 0
                for r in rewards:
                    idx = self.current_subpolicy[cursor]
                    subpolicy_minibatch[idx][MLSHRunner.MB_REWARDS].append(r)
                    cursor += 1




        return mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, mb_states, ep_infos, true_reward

    def subpolicy_step(self, obs, states, done):
        cursor = 0
        actions = []
        values = []
        states = []
        neglogpacs = []

        # for each subpolicy, compute action, values, and negative log probabilities
        for o,s,d in zip(obs, states, done):
            subpolicy = self.current_subpolicy[cursor]
            a, v, new_state, negp = subpolicy.step(o, s, d)
            actions.append(a)
            values.append(v)
            states.append(new_state)
            neglogpacs.append(negp)
            cursor += 1

        return actions, values, states, neglogpacs

    def to_subpolicy_mnbatch(self, minibatch):
        cursor = 0
        subpolicy_minibatch = {}
        for i in range(len(self.subpolicies)):
            subpolicy_minibatch[i] = []
        for mb_obs, mb_rewards, mb_actions, mb_values, mb_neglopacs, mb_dones in zip(minibatch):
            subpolicy_index = self.current_subpolicy[cursor]
            subpolicy_minibatch[subpolicy_index].append((mb_obs, mb_rewards, mb_actions, mb_values, mb_neglopacs, mb_dones))
        return subpolicy_minibatch





def get_temp_env(observation_space, num_subpolicy):
    class TempEnv(gym.Env):
        def __init__(self):
            self.observation_space = observation_space
            self.action_space = gym.spaces.Discrete(num_subpolicy)

        def step(self, action):
            pass

        def reset(self):
            pass

        def render(self, mode='human'):
            pass
    return TempEnv