from mlsh import MLSH
from mlsh import MLSHSubpolicyDQN
from mlsh import MLSHSubpolicySAC

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.sac.policies import LnMlpPolicy
import gym
from stable_baselines import TRPO
import os

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    env = gym.make("BipedalWalker-v2")

    model = MLSH(policy=MlpPolicy, env=env, subpolicy_network=LnMlpPolicy, num_subpolicies=3,
                 verbose=1, timesteps_per_batch=128, subpolicy_time=24)

    model = MLSH.load("BipedalWalker_MLSH", num_subpolicy=3)
    model.setup_env(env)
    for t in range(100):

        model.learn(2000)
        model.save("BipedalWalker_MLSH")
        print("reset master")
        model.reset_master()
        if t >= 3:
            model.optimize_subpolicy(1000)