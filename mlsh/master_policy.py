import gym
import tensorflow as tf
from collections import deque
from stable_baselines.trpo_mpi.trpo_mpi import TRPO, traj_segment_generator
from mlsh_training_env_wrapper import MLSHEnvWraper
from sub_policy import MLSHSubpolicy

class MLSH(TRPO):
    def __init__(self, policy, env, num_subpolicy=3, subpolicy_time=50, warmup=20, gamma=0.99, timesteps_per_batch=12, max_kl=0.01, cg_iters=10, lam=0.98,
                 entcoeff=0.0, cg_damping=1e-2, vf_stepsize=3e-4, vf_iters=3, verbose=0, tensorboard_log=None,
                 _init_setup_model=True, policy_kwargs=None, full_tensorboard_log=False,
                 seed=None, n_cpu_tf_sess=1):
        self.subpolicies = [MLSHSubpolicy(env=env, policy=policy, timesteps_per_batch=1024, verbose=0) for _ in range(num_subpolicy)]
        wrapped_env = MLSHEnvWraper(env=env, subpolicies=self.subpolicies, subpolicy_time_schedule=subpolicy_time, warmup=warmup, gamma=gamma)
        super(MLSH, self).__init__(policy=policy, env=wrapped_env, verbose=verbose, timesteps_per_batch=timesteps_per_batch,
                                   max_kl=max_kl, lam=lam, entcoeff=entcoeff, cg_damping=cg_damping,
                                   vf_stepsize=vf_stepsize, vf_iters=vf_iters, tensorboard_log=tensorboard_log,
                                   _init_setup_model=_init_setup_model, policy_kwargs=policy_kwargs,
                                   full_tensorboard_log=full_tensorboard_log, seed=seed, n_cpu_tf_sess=n_cpu_tf_sess,
                                            gamma=gamma, cg_iters=cg_iters)

    def setup_model(self):
        self.env.warmup_cnt = 0
        return super().setup_model()

    def subpolicy_warmup(self, steps):
        for s in self.subpolicies:
            s.learn(steps)
            print("--subpolicy warm up --")
import gym
import test_envs
from stable_baselines.common.policies import MlpPolicy

class wrappedenv(gym.Env):
    def __init__(self, env):
        self.env =env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.episode_reward = 0

    def reset(self):
        print(self.episode_reward)
        self.episode_reward = 0
        return self.env.reset()

    def step(self, action):
        s, r, d, i = self.env.step(action)
        self.episode_reward += r
        return s, r, d, i
if __name__ =="__main__":
    env = wrappedenv(gym.make("MovementBandits-v0"))
    model = TRPO(env=env, policy=MlpPolicy, timesteps_per_batch=128)
    model.learn(50000 * 8)
    """
    model = MLSH(env=env, policy=MlpPolicy, verbose=0, subpolicy_time=256, vf_stepsize=1e-3)
    print("warm up start")
    model.subpolicy_warmup(20000)
    print("warm up done ")
    for i in range(10):
        model.learn(100, log_interval=10)
        model.setup_model()
        print("reset master")
    model.learn(10000)
    """