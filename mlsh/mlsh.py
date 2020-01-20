import time
import numpy as np
import gym
from collections import deque

from stable_baselines.common import SetVerbosity, TensorboardWriter, explained_variance, make_vec_env
from stable_baselines.common.policies import RecurrentActorCriticPolicy
from stable_baselines.common.base_class import BaseRLModel, _UnvecWrapper
from stable_baselines.common.runners import AbstractEnvRunner

from stable_baselines.ppo2.ppo2 import swap_and_flatten, get_schedule_fn, safe_mean
from stable_baselines.ppo2.ppo2 import Runner as PPORunner
from stable_baselines.ppo2 import PPO2

from stable_baselines import logger
from stable_baselines.a2c.utils import total_episode_reward_logger


class MLSH(BaseRLModel):
    def __init__(self, master_policy, env, subpolicy_timestep=200, num_subpolicy=3,
                 master_lr=1e-2, subpolicy_lr=3e-4, tensorboard_log=None, gamma=0.99, lam=0.95, verbose=0,
                 subpolicy_model_policy=None, warm_up=2, *args, **kwargs):
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
                                    n_envs=env.num_envs)
        self.master_lr = master_lr
        self.master_policy_arch = master_policy
        self.verbose = verbose
        self.master_model = PPO2(env=self.tempenv, policy=self.master_policy_arch, learning_rate=self.master_lr, verbose=self.verbose)
        self.warm_up = warm_up

        self.subpolicy_model_learning_rate = subpolicy_lr
        self.tensorboard_log=tensorboard_log
        self.gamma = gamma
        self.subpolicies =[]
        self.subpolicy_runners = []
        self.subpolicy_timestep = subpolicy_timestep
        self.current_subpolicy = 0
        self.lam = lam
        self.ep_info_buf = deque(maxlen=100)


        for i in range(num_subpolicy):
            self.subpolicies.append(PPO2(subpolicy_model_policy, env))
        for i in range(num_subpolicy):
            self.subpolicy_runners.append(PPORunner(env= self.env, model=self.subpolicies[i],
                                                    n_steps=self.subpolicy_timestep, gamma=self.gamma, lam=self.lam))

    def setup_model(self):
        self.master_model.setup_model()
        for s in self.subpolicies:
            s.setup_model()

    def _get_pretrain_placeholders(self):
        master_policy = self.master_model.act_model
        return master_policy.obs_ph, self.master_model.action_ph, master_policy.policy

    def get_parameter_list(self):
        pass

    def master_action_probability(self, observation, state=None, mask=None, actions=None, logp=False):
        return self.master_model.action_probability(observation, state, mask, actions, logp)

    def subpolicy_action_probability(self, subpolicy_index, observation, state=None, mask=None, actions=None, logp=False):
        return self.subpolicies[subpolicy_index].action_probability\
            (observation, state, mask, actions, logp)

    def action_probability(self, observation, state=None, mask=None, actions=None, logp=False):
        return self.subpolicy_action_probability(subpolicy_index=self.current_subpolicy, observation=observation,
                                                 state=state, mask=mask, actions=actions, logp=logp)

    def learn(self, total_timesteps, callback=None, master_log_interval=5, subpolicy_log_interval=2, tb_log_name="run",
              reset_num_timesteps=True):

        new_tb_log = self._init_num_timesteps(reset_num_timesteps)

        with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) \
                as writer:

            # initialize theta (reset master model)
            self.master_model = PPO2(env=self.tempenv, policy=self.master_policy_arch, learning_rate=self.master_lr)
            self.master_model._setup_learn()
            learning_rate = get_schedule_fn(self.master_model.learning_rate)
            cliprange = get_schedule_fn(self.master_model.cliprange)
            cliprange_vf = get_schedule_fn(self.master_model.cliprange_vf)

            subpolicy_scheduler = [[get_schedule_fn(s.learning_rate), get_schedule_fn(s.cliprange),
                                                    get_schedule_fn(s.cliprange_vf)] for s in self.subpolicies]

            self.master_model._setup_learn()
            runner = MLSHRunner(env=self.env, gamma = self.gamma, lam = self.lam,
                                n_steps=self.master_model.n_steps, master_model=self.master_model,
                                subpolicies=self.subpolicies, subpolicy_timestep=self.subpolicy_timestep)

            t_first_start = time.time()
            n_batch = self.master_model.n_batch

            n_updates = total_timesteps // self.master_model.n_batch
            for update in range(1, n_updates + 1):
                batch_size = n_batch // self.master_model.nminibatches
                t_start = time.time()
                frac = 1.0 - (update - 1.0) / n_updates
                lr_now =  learning_rate(frac)
                cliprange_now = cliprange(frac)
                cliprange_vf_now = cliprange_vf(frac)

                # for warm up do not collect subpolicy trajectories
                obs, returns, masks, actions, values, neglogpacs, states, ep_infos, true_reward, subpolicy_traj  = runner.run()

                # for memory efficiency, delete the objects
                if update < self.warm_up:
                    del subpolicy_traj
                else:
                    # for each sub policy trajectory
                    for j in range(len(self.subpolicies)):
                        # for each subpolicies
                        target_subpolicy = self.subpolicies[j]
                        subpolicy_batch =[]
                        frac_s = 1.0 - ((update - self.warm_up)  - 1.0) / (n_updates - self.warm_up)
                        lr_now_s = subpolicy_scheduler[j][0](frac_s)
                        cliprange_now_s = subpolicy_scheduler[j][1](frac_s)
                        cliprange_vf_now_s = subpolicy_scheduler[j][2](frac_s)

                        for i in range(len(subpolicy_traj)):
                            subpoliciy_mnbatch = subpolicy_traj[i][j]
                            if not subpoliciy_mnbatch.valid:
                                continue
                            else:
                                subpolicy_batch.append(subpoliciy_mnbatch)

                        # merge all transitions to batch
                        obs_s = []
                        returns_s = []
                        masks_s = []
                        actions_s = []
                        values_s = []
                        neglogpacs_s = []
                        true_rewards_s = []
                        for i in range(len(subpolicy_batch)):
                            subpoliciy_mnbatch = subpolicy_batch[i]
                            obs_s.append(subpoliciy_mnbatch["obs"])
                            returns_s.append(subpoliciy_mnbatch["returns"])
                            masks_s.append(subpoliciy_mnbatch["dones"])
                            actions_s.append(subpoliciy_mnbatch["actions"])
                            values_s.append(subpoliciy_mnbatch["values"])
                            neglogpacs_s.append(subpoliciy_mnbatch["neglogpacs"])
                            true_rewards_s.append(subpoliciy_mnbatch["true_rewards"])

                        obs_s = np.vstack(obs_s)
                        actions_s = np.squeeze(np.vstack(actions_s))
                        masks_s = np.hstack(masks_s)
                        values_s = np.hstack(values_s)
                        returns_s = np.hstack(returns_s)
                        neglogpacs_s = np.squeeze(np.vstack(neglogpacs_s))
                        true_rewards_s = np.hstack(true_rewards_s)
                        states_s = None
                        # states_s = subpoliciy_mnbatch["states"]  # This is for recurrent policy,
                        update_fac = target_subpolicy.n_batch // target_subpolicy.nminibatches // target_subpolicy.noptepochs + 1
                        inds = np.arange(len(returns_s))

                        mb_loss_vals_s = []
                        for epoch_num in range(target_subpolicy.noptepochs):
                            np.random.shuffle(inds)
                            for start in range(0, target_subpolicy.n_batch, batch_size):
                                timestep_s = self.num_timesteps // update_fac + ((target_subpolicy.noptepochs * target_subpolicy.n_batch
                                                                                  + epoch_num
                                                                                  * target_subpolicy.n_batch + start) // batch_size)
                                end = start + batch_size
                                mbinds = inds[start:end]
                                slices = (arr[mbinds] for arr in (obs_s, returns_s, masks_s, actions_s, values_s, neglogpacs_s))
                                mb_loss_vals_s.append(target_subpolicy._train_step(lr_now_s, cliprange_now_s, *slices, writer=writer,
                                                                     update=timestep_s, cliprange_vf=cliprange_vf_now_s))

                        loss_vals_s = np.mean(mb_loss_vals_s, axis=0)
                        if self.verbose >= 1 and (update % subpolicy_log_interval == 0 or update == 1):
                            explained_var_s = explained_variance(values_s, returns_s)
                            logger.logkv("subpolicy index", j)
                            logger.logkv("explained_variance", float(explained_var_s))
                            logger.logkv("loss", safe_mean(np.asarray(mb_loss_vals_s, dtype=np.float32)))
                            for (loss_val, loss_name) in zip(loss_vals_s, target_subpolicy.loss_names):
                                logger.logkv(loss_name, loss_val)
                            logger.dumpkvs()
                self.num_timesteps += 1
                self.ep_info_buf.extend(ep_infos)
                mb_loss_vals = []

                update_fac = self.master_model.n_batch // self.master_model.nminibatches // self.master_model.noptepochs + 1
                inds = np.arange(self.master_model.n_batch)
                for epoch_num in range(self.master_model.noptepochs):
                    np.random.shuffle(inds)
                    for start in range(0, self.master_model.n_batch, batch_size):
                        timestep = self.num_timesteps // update_fac + ((self.master_model.noptepochs * self.master_model.n_batch + epoch_num *
                                                                        self.master_model.n_batch + start) // batch_size)
                        end = start + batch_size
                        mbinds = inds[start:end]
                        slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                        mb_loss_vals.append(self.master_model._train_step(lr_now, cliprange_now, *slices, writer=writer,
                                                             update=timestep, cliprange_vf=cliprange_vf_now))

                loss_vals = np.mean(mb_loss_vals, axis=0)
                t_now = time.time()
                fps = int(self.master_model.n_batch / (t_now - t_start))


                if writer is not None:
                    total_episode_reward_logger(self.master_model.episode_reward,
                                                true_reward.reshape((self.n_envs, self.master_model.n_steps)),
                                                masks.reshape((self.n_envs, self.master_model.n_steps)),
                                                writer, self.num_timesteps)

                if self.verbose >= 1 and (update % master_log_interval == 0 or update == 1):
                    explained_var = explained_variance(values, returns)
                    logger.logkv("serial_timesteps", update * self.master_model.n_steps)
                    logger.logkv("n_updates", update)
                    logger.logkv("total_timesteps", self.num_timesteps)
                    logger.logkv("fps", fps)
                    logger.logkv("explained_variance", float(explained_var))
                    if len(self.ep_info_buf) > 0 and len(self.ep_info_buf[0]) > 0:
                        logger.logkv('ep_reward_mean', safe_mean([ep_info['r'] for ep_info in self.ep_info_buf]))
                        logger.logkv('ep_len_mean', safe_mean([ep_info['l'] for ep_info in self.ep_info_buf]))
                    logger.logkv('time_elapsed', t_start - t_first_start)
                    for (loss_val, loss_name) in zip(loss_vals, self.master_model.loss_names):
                        logger.logkv(loss_name, loss_val)
                    logger.dumpkvs()

                if callback is not None:
                    # Only stop training if return value is False, not when it is None. This is for backwards
                    # compatibility with callbacks that have no return statement.
                    if callback(locals(), globals()) is False:
                        break

        return self

    @classmethod
    def load(cls, load_path, env=None, custom_objects=None, **kwargs):
        pass

    def save(self, save_path, cloudpickle=False):
        pass

    def predict(self, observation, state=None, mask=None, deterministic=False):
        return self.subpolicies[self.current_subpolicy].predict(observation, state, mask, deterministic)

    def predict_master(self, observation, state=None, mask=None, deterministic=False):
        return self.master_model.predict(observation, state, mask, deterministic)


class SubpolicyMinibatch(dict):
    """
    Container objects for minibatches of subpolicies
    """
    _key = ("obs", "dones","rewards", "actions", "values", "neglogpacs",  "last_values")
    # key of data whose type is np.float32 and requires casting
    _float_key = _key[2: ]

    def __init__(self):
        super(SubpolicyMinibatch, self).__init__()
        for k in SubpolicyMinibatch._key:
            self[k] = []
        self.valid = False

    @classmethod
    def new_mnbatches(cls, num_subpolicies):
        return [cls() for _ in range(num_subpolicies)]

    @staticmethod
    def to_batch(mnbatch_list, obs_dtype, gamma, lam, last_values, n_steps):
        # batch of steps to batch of roll outs
        for i in range(len(mnbatch_list)):
            mnbatch = mnbatch_list[i]
            # the subpolicy has not chosen
            if len(mnbatch["dones"]) == 0:
                continue
            mnbatch.valid = True
            for k in SubpolicyMinibatch._float_key:
                data = mnbatch[k]
                mnbatch[k] = np.asarray(data, dtype=np.float32)
            mnbatch["obs"] = np.asarray(mnbatch["obs"], dtype=obs_dtype)
            mnbatch["dones"] = np.asarray(mnbatch["dones"], dtype=np.bool)
            mb_rewards = mnbatch["rewards"]
            mb_values = mnbatch["values"]
            mb_dones = mnbatch["dones"]
            mb_advs = np.zeros_like(mb_rewards)
            true_reward = np.copy(mb_rewards)

            last_gae_lam = 0
            for step in reversed(range(n_steps)):
                if step == n_steps - 1:
                    nextnonterminal = 1.0 - mb_dones[-1]
                    nextvalues = last_values[i]

                else:
                    nextnonterminal = 1.0 - mb_dones[step + 1]
                    nextvalues = mb_advs[step + 1]

                # computes advantage
                delta = mb_rewards[step] + gamma * nextvalues * nextnonterminal - mb_values[step]
                mb_advs[step] = last_gae_lam = delta + gamma * lam * nextnonterminal * last_gae_lam

            # computes returns

            mb_values = mb_values.flatten()
            mb_returns = mb_advs + mb_values

            mnbatch["values"] = mb_values
            mnbatch["true_rewards"] = true_reward
            mnbatch["returns"] = mb_returns

        return mnbatch_list




class MLSHRunner(AbstractEnvRunner):

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
        subpolicy_transitions =[]

        # Roll out master policy
        for _ in range(self.n_steps):
            actions, values, self.states, neglogpacs = self.model.step(self.obs, self.states, self.dones)
            master_obs.append(self.obs.copy())
            master_actions.append(actions)
            master_values.append(values)
            master_neglogpacs.append(neglogpacs)
            master_dones.append(self.dones)
            clipped_actions = actions

            master_reward_step =[]
            # Choose subpolicy
            self.current_subpolicy = actions

            # roll out subpolicy
            subpolicy_minibatch = SubpolicyMinibatch.new_mnbatches(len(self.subpolicies))

            for __ in range(self.subpolicy_timestep):
                subpolicy_actions, subpolicy_values, states, subpolicy_neglopcacs =self.subpolicy_step(self.obs, self.states, self.dones)
                cursor = 0

                # for each subpolicy, append transition results to its own minibatches
                for o, a, v, n, d in zip(self.obs, subpolicy_actions, subpolicy_values, subpolicy_neglopcacs, self.dones):
                    idx = self.current_subpolicy[cursor]
                    subpolicy_minibatch[idx]["obs"].append(o.copy())
                    subpolicy_minibatch[idx]["actions"].append(a)
                    subpolicy_minibatch[idx]["values"].append(v)
                    subpolicy_minibatch[idx]["neglogpacs"].append(n)
                    subpolicy_minibatch[idx]["dones"].append(d)
                    cursor += 1

                # clip actions for each subpolicies
                if isinstance(self.env.action_space, gym.spaces.Box):
                    subpolicy_actions_step = np.asarray(subpolicy_actions, dtype=np.float32)
                    clipped_actions = np.clip(subpolicy_actions_step, self.env.action_space.low, self.env.action_space.high)
                else:
                    clipped_actions = np.asarray(subpolicy_actions, dtype=np.int64).flatten()
                # sync-steps for each environments
                self.obs[:], rewards, self.dones, infos = self.env.step(clipped_actions)

                # to compute master rewards, keep the reward results in the separate lists
                master_reward_step.append(np.copy(rewards))

                # note that info is collected in the subpolicy level.
                for info in infos:
                    maybe_ep_info = info.get('episode')
                    if maybe_ep_info is not None:
                        ep_infos.append(maybe_ep_info)

                # for each subpolicy, append rewards
                cursor = 0
                for r in rewards:
                    idx = self.current_subpolicy[cursor]
                    subpolicy_minibatch[idx]["rewards"].append(r)
                    cursor += 1
            last_values = self.subpolicy_values(self.obs, self.states, self.dones)

            # Now Keep mini-batches in a lists.
            # In the paper, subpolicies are trained after the master has been trained. We have to keep the mini-batches
            # until we finish up collecting all master transition data
            subpolicy_minibatch = SubpolicyMinibatch.to_batch(subpolicy_minibatch, obs_dtype=self.obs.dtype,
                                                              gamma=self.gamma, lam=self.lam, last_values=last_values,
                                                              n_steps=self.subpolicy_timestep)

            subpolicy_transitions.append(subpolicy_minibatch)

            master_reward_step = np.asarray(master_reward_step)

            # Time scale is 1/N. Sum up the rewards of the master's steps

            master_reward_step = np.sum(master_reward_step, axis=0)
            master_rewards.append(master_reward_step)

        # Now everything is same as original ppo algorithm

        master_obs = np.asarray(master_obs, dtype=self.obs.dtype)
        master_rewards = np.asarray(master_rewards, dtype=np.float32)
        master_actions = np.asarray(master_actions)
        master_values = np.asarray(master_values, dtype=np.float32)
        master_neglogpacs = np.asarray(master_neglogpacs, dtype=np.float32)
        master_dones = np.asarray(master_dones, dtype=np.bool)
        last_values = self.master_model.value(self.obs, self.states, self.dones)
        # discount/bootstrap off value fn
        master_advs = np.zeros_like(master_rewards)
        true_reward = np.copy(master_rewards)
        last_gae_lam = 0
        for step in reversed(range(self.n_steps)):
            if step == self.n_steps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - master_dones[step + 1]
                nextvalues = master_values[step + 1]
            delta = master_rewards[step] + self.gamma * nextvalues * nextnonterminal - master_values[step]
            master_advs[step] = last_gae_lam = delta + self.gamma * self.lam * nextnonterminal * last_gae_lam

        master_returns = master_advs + master_values

        master_obs, master_returns, master_dones, master_actions, master_values, master_neglogpacs, true_reward = \
            map(swap_and_flatten, (master_obs, master_returns, master_dones, master_actions, master_values, master_neglogpacs, true_reward))


        return master_obs, master_returns, master_dones, master_actions, master_values, master_neglogpacs, mb_states, ep_infos, true_reward, subpolicy_transitions

    def subpolicy_step(self, obs, states, done):
        cursor = 0
        actions = []
        values = []
        states_ = []
        neglogpacs = []

        # for each subpolicy, compute action, values, and negative log probabilities
        if states is not None:
            for o,s,d in zip(obs, states, done):
                subpolicy = self.current_subpolicy[cursor]
                o = np.expand_dims(o, axis=0)
                a, v, new_state, negp = subpolicy.step(o, s, d)
                actions.append(a)
                values.append(v)
                states_.append(new_state)
                neglogpacs.append(negp)
                cursor += 1
        else:
            for o, d in zip(obs, done):
                subpolicy_index = self.current_subpolicy[cursor]
                o = np.expand_dims(o, axis=0)
                subpolicy = self.subpolicies[subpolicy_index]
                a, v, new_state, negp = subpolicy.step(o, None , d)
                actions.append(a)
                values.append(v)
                states_.append(new_state)
                neglogpacs.append(negp)
                cursor += 1

        return actions, values, None, neglogpacs

    def subpolicy_values(self, obs, states, done):
        cursor = 0
        values = []
        # for each subpolicy, compute action, values, and negative log probabilities
        if states is not None:
            for o,s,d in zip(obs, states, done):
                o = np.expand_dims(o, axis=0)
                subpolicy = self.current_subpolicy[cursor]
                v = subpolicy.value(o, s, d)
                values.append(v)
                cursor += 1
        else:
            for o, d in zip(obs, done):
                subpolicy_index = self.current_subpolicy[cursor]
                o = np.expand_dims(o, axis=0)
                subpolicy = self.subpolicies[subpolicy_index]
                v = subpolicy.value(o, None , d)
                values.append(v)
                cursor += 1

        values = np.asarray(values, dtype=np.float32)
        values = values.flatten()
        return values

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


from stable_baselines.common.policies import MlpPolicy
if __name__ == "__main__":
    env = make_vec_env("CartPole-v0", n_envs=4)
    model = MLSH(env=env, master_policy=MlpPolicy, subpolicy_timestep=10, num_subpolicy=2, verbose=1)
    model.learn(10000000, master_log_interval=10, subpolicy_log_interval=1)