import gym
import tensorflow as tf
from collections import deque
from stable_baselines.trpo_mpi.trpo_mpi import TRPO, traj_segment_generator
from mlsh_training_env_wrapper import MLSHEnvWraper
from sub_policy import MLSHSubpolicy


class MLSH(TRPO):
    def __init__(self, policy, env, num_subpolicy=3, subpolicy_time=50, warmup=20, gamma=0.99, timesteps_per_batch=4, max_kl=0.01, cg_iters=10, lam=0.98,
                 entcoeff=0.0, cg_damping=1e-2, vf_stepsize=3e-4, vf_iters=3, verbose=0, tensorboard_log=None,
                 _init_setup_model=True, policy_kwargs=None, full_tensorboard_log=False,
                 seed=None, n_cpu_tf_sess=1):
        self.subpolicies = [MLSHSubpolicy(env=env, policy=policy, timesteps_per_batch=subpolicy_time, verbose=0) for _ in range(num_subpolicy)]
        wrapped_env = MLSHEnvWraper(env=env, subpolicies=self.subpolicies, subpolicy_time_schedule=subpolicy_time, warmup=warmup, gamma=gamma)
        super(MLSH, self).__init__(policy=policy, env=wrapped_env, verbose=verbose, timesteps_per_batch=timesteps_per_batch,
                                   max_kl=max_kl, lam=lam, entcoeff=entcoeff, cg_damping=cg_damping,
                                   vf_stepsize=vf_stepsize, vf_iters=vf_iters, tensorboard_log=tensorboard_log,
                                   _init_setup_model=_init_setup_model, policy_kwargs=policy_kwargs,
                                   full_tensorboard_log=full_tensorboard_log, seed=seed, n_cpu_tf_sess=n_cpu_tf_sess,
                                            gamma=gamma, cg_iters=cg_iters)


import gym
from stable_baselines.common.policies import MlpPolicy
if __name__ =="__main__":
    env = gym.make("CartPole-v0")
    model = MLSH(env=env, policy=MlpPolicy, verbose=0, subpolicy_time=64, vf_stepsize=1e-2)
    for i in range(100):
        model.learn(1000)
        model.setup_model()
        print("reset master")