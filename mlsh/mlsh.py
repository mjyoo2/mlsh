import gym
import os

from stable_baselines import TRPO
from mlsh.mlsh_training_env_wrapper import MLSHEnvWraper
from mlsh.sub_policy import MLSHSubpolicySAC, MLSHSubpolicyDQN

from stable_baselines.common.vec_env import DummyVecEnv
from collections import Iterable


class MLSH(TRPO):
    """
    MLSH (master) policy.
    """
    def __init__(self, policy, env, subpolicy_network, num_subpolicies,
                 subpolicies=None,
                 subpolicy_kwargs=None,
                 subpolicy_time=50, warmup=20,gamma=0.99,  verbose=0,
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
        wrapped_env =DummyVecEnv([lambda: MLSHEnvWraper(env=env, subpolicies=self.subpolicies,
                                                        subpolicy_time_schedule=subpolicy_time,
                                                        warmup=warmup, gamma=gamma)])

        super(MLSH, self).__init__(policy=policy, env=wrapped_env, verbose=verbose,
                                   *args, **kwargs)

    def setup_model(self):
        self.env.warmup_cnt = 0
        return super().setup_model()

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
        os.mkdir(save_dir)

        subpolicy_name = save_dir + "/" + subpolicy_name + "{}"
        master_name = save_dir + "/master"

        for i in range(len(self.subpolicies)):
            s = self.subpolicies[i]
            s.save(save_path=subpolicy_name.format(i), cloudpickle=cloudpickle)
        return super().save(save_path=master_name, cloudpickle=cloudpickle)

    @classmethod
    def load(cls, load_path, discrete=False, env=None, custom_objects=None,
             master_name="master", subpolicy_name="subpolicy{}",
             num_subpolicy=None, env_kwargs=None, **kwargs):


        if discrete:
            subpolicy_type = MLSHSubpolicyDQN
        else:
            subpolicy_type = MLSHSubpolicySAC

        subpolicies = None
        if isinstance(subpolicy_name, str):
            assert num_subpolicy is not None, "You are passing string args for subpolicy name. " \
                                              "you have to give the number of subpolicies"
            subpolicies = [subpolicy_type.load(subpolicy_name.format(i), env=env, custom_objects=custom_objects) for i
             in range(num_subpolicy)]
        elif isinstance(subpolicy_name, Iterable):
            subpolicies = [subpolicy_type.load(s_name, env=env, custom_objects=custom_objects) for s_name in subpolicy_name]

        master_name = load_path + "/" + master_name
        master = super(MLSH, cls).load(master_name, env=env, custom_objects=custom_objects)
        master.subpolicies = subpolicies
        return master