import gym
import os
import shutil

from stable_baselines import TRPO
from mlsh_training_env_wrapper import MLSHEnvWraper
from sub_policy import MLSHSubpolicySAC, MLSHSubpolicyDQN

from collections import Iterable
from stable_baselines.common import ActorCriticRLModel


class MLSH(TRPO):
    """
    MLSH (master) policy.
    """

    def __init__(self, policy, env, subpolicy_network=None, num_subpolicies=None,
                 subpolicies=None,
                 subpolicy_kwargs=None,
                 subpolicy_time=50, warmup=1000,gamma=0.99,  verbose=0, _init_setup_model=True,
                 *args, **kwargs):
        """
        :param policy: policy networks for master policy. This name may be confusing, but kept to be consistent with
        stable baseline keyward
        :param env: environment to learn. NOTE THAT MASTER POLICY WILL HAVE WRAPPED ENV.
        :param subpolicy_network: subpolicies' network. If you are using Box action space, it should be SAC's policy
        otherwise it should be DQN's policy
        :param num_subpolicies: number of subpolicy to choose
        :param subpolicies: if you want to use subpolicy already created, you may pass the subpolicy as a lists
        :param subpolicy_kwargs: keyward arguments for subpolicy networks
        :param subpolicy_time: how many time steps do you want to run for each subpolicy ?
        :param warmup: warm up time step; MAY be changed into warm up episodes
        :param gamma: discount factor
        :param verbose: verbose
        :param args: master policy's args
        :param kwargs: master policy's kwargs
        """
        self.env = None
        self.policy = policy
        self.subpolicy_time = None
        self.warmup = None
        self.master_args = args
        self.master_kwargs = kwargs
        self.wrapping_env = None

        if _init_setup_model:
            if subpolicies is None:
                if subpolicy_kwargs is None:
                    subpolicy_kwargs = {}
                if isinstance(env.action_space, gym.spaces.Box):
                    self.subpolicies = [MLSHSubpolicySAC(env= env, policy=subpolicy_network,
                                                         **subpolicy_kwargs) for _ in range(num_subpolicies)]
                elif isinstance(env.action_space, gym.spaces.Discrete):
                    self.subpolicies = [MLSHSubpolicyDQN(env=env, policy=subpolicy_network,
                                                         **subpolicy_kwargs) for _ in range(num_subpolicies)]
                    Warning("Discrete policy may not work yet!! it is not complete... ")
                self.subpolicies[0].name = "debugger"

            else:
                self.subpolicies = subpolicies
                if subpolicy_kwargs is not None:
                    Warning("You already passed created subpolicies, your subpolicy kwargs will be ignored")

            self.num_subpolicies = len(self.subpolicies)
            self.subpolicy_time = subpolicy_time
            self.warmup = warmup
            self.wrapping_env = MLSHEnvWraper(env=env, subpolicies=self.subpolicies,
                                                                   subpolicy_time_schedule=subpolicy_time,
                                                                   warmup=warmup, gamma=gamma)
            for s in self.subpolicies:
                s.setup_learn()

            super(MLSH, self).__init__(policy=self.policy, env=self.wrapping_env, verbose=verbose,
                                       *args, **kwargs)
        else:
            kwargs["_init_setup_model"] = False
            super(MLSH, self).__init__(policy=policy, env=env, verbose=verbose,
                                       *args, **kwargs)

    def master_initializer(self):
        env = MLSHEnvWraper(env=self.wrapped_env, subpolicies=self.subpolicies,
                                           subpolicy_time_schedule=self.subpolicy_time,
                                           warmup=self.warmup, gamma=self.gamma)
        print(self.master_kwargs)
        print(self.master_args)
        super(MLSH, self).__init__(policy=self.policy, env=env,
                                            verbose=self.verbose, _init_setup_model=True,
                                            *self.master_args, **self.master_kwargs)

    def setup_env(self, env):
        wrapping_env = MLSHEnvWraper(env=env, subpolicies=self.subpolicies,
                                                         subpolicy_time_schedule=self.subpolicy_time,
                                                         warmup=self.warmup, gamma=self.gamma)
        for s in self.subpolicies:
            s.set_env(env)
        self.env = wrapping_env

    @property
    def wrapped_env(self):
        if self.env is None:
            return None
        else:
            return self.env.wrapped_env

    def learn(self, total_timesteps, callback=None, log_interval=100, tb_log_name="MLSH",
              reset_num_timesteps=True):
        self.env.warmup_cnt = 0
        return super().learn(total_timesteps=total_timesteps, callback=callback,
                                       log_interval=log_interval, tb_log_name=tb_log_name,
                                             reset_num_timesteps=reset_num_timesteps)

    def optimize_subpolicy(self, steps=5000) -> None:
        """
        Optimize subpolicy out sides of learning step. You have to check subpolicy has sufficiently many
        replay buffer!
        :param steps: number of steps to learn
        :return: None
        """
        for s in self.subpolicies:
            for _ in range(steps):
                s.learn_async()
            s.async_step += steps

    def warmup_subpolicy(self, steps=10000, verbose=1) -> None:
        """
        :param steps: warm up subpolicy.
        NOTE THAT This function calls "learn" method
        :param verbose: verbosity of learn function
        :return:
        """
        for s in self.subpolicies:
            verbosity = s.verbose
            s.verbose= verbose
            s.learn(steps)
            s.async_step += steps
            s.verbose = verbosity

    def clear_replay_buffer(self):
        """
        clear subpolicies' replay buffer
        :return:
        """
        for s in self.subpolicies:
            s.clear_replay_buffer()

    def save(self, save_path, cloudpickle=False, master_name="master", subpolicy_name="subpolicy"):
        save_dir = save_path
        try:
            os.mkdir(save_dir)
        except FileExistsError:
            shutil.rmtree(save_dir, ignore_errors=True)
            os.mkdir(save_dir)

        subpolicy_name = save_dir + "/" + subpolicy_name + "{}"
        master_name = save_dir + "/master"

        for i in range(len(self.subpolicies)):
            s = self.subpolicies[i]
            s.save(save_path=subpolicy_name.format(i), cloudpickle=cloudpickle)

        data = {
            "gamma": self.gamma,
            "timesteps_per_batch": self.timesteps_per_batch,
            "max_kl": self.max_kl,
            "cg_iters": self.cg_iters,
            "lam": self.lam,
            "entcoeff": self.entcoeff,
            "cg_damping": self.cg_damping,
            "vf_stepsize": self.vf_stepsize,
            "vf_iters": self.vf_iters,
            "hidden_size_adversary": self.hidden_size_adversary,
            "adversary_entcoeff": self.adversary_entcoeff,
            "expert_dataset": self.expert_dataset,
            "g_step": self.g_step,
            "d_step": self.d_step,
            "d_stepsize": self.d_stepsize,
            "using_gail": self.using_gail,
            "verbose": self.verbose,
            "policy": self.policy,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "n_envs": self.n_envs,
            "n_cpu_tf_sess": self.n_cpu_tf_sess,
            "seed": self.seed,
            "_vectorize_action": self._vectorize_action,
            "warmup": self.warmup,
            "subpolicy_time": self.subpolicy_time,
            "policy_kwargs": self.policy_kwargs
        }

        params_to_save = self.get_parameters()
        self._save_to_file(master_name, data=data, params=params_to_save, cloudpickle=cloudpickle)


    @classmethod
    def load(cls, load_path, discrete=False, env=None, custom_objects=None,
             master_name="master", subpolicy_name="subpolicy{}",
             num_subpolicy=None, env_kwargs=None, suffix=".zip", **kwargs):

        if discrete:
            subpolicy_type = MLSHSubpolicyDQN
        else:
            subpolicy_type = MLSHSubpolicySAC

        subpolicies = None
        if isinstance(subpolicy_name, str):
            assert num_subpolicy is not None, "You are passing string args for subpolicy name. " \
                                              "you have to give the number of subpolicies"
            subpolicies = [subpolicy_type.load(load_path + "/" + subpolicy_name.format(i) + suffix,
                                               env=env, custom_objects=custom_objects) for i in range(num_subpolicy)]
        elif isinstance(subpolicy_name, Iterable):
            subpolicies = [subpolicy_type.load(load_path + "/" + s_name, env=env, custom_objects=custom_objects)
                                                                                        for s_name in subpolicy_name]

        master_name = load_path + "/" + master_name

        master_data, master_param = cls._load_from_file(master_name + suffix, custom_objects=custom_objects)
        master_data["subpolicies"] = subpolicies
        if 'policy_kwargs' in kwargs and kwargs['policy_kwargs'] != master_data['policy_kwargs']:
            raise ValueError("The specified policy kwargs do not equal the stored policy kwargs. "
                             "Stored kwargs: {}, specified kwargs: {}".format(master_data['policy_kwargs'],
                                                                              kwargs['policy_kwargs']))
        model = cls(policy=master_data["policy"], env=None, _init_setup_model=False, subpolicies=master_data["subpolicies"])
        model.__dict__.update(master_data)
        model.__dict__.update(kwargs)
        if env is not None:
            model.setup_env(env)
        model.setup_model()

        return model

    def reset_master(self):
        self.setup_model()