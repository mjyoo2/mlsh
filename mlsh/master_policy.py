from stable_baselines.common.base_class import BaseRLModel, _UnvecWrapper
from gym.spaces import Discrete
from stable_baselines.common.vec_env import VecEnv
from stable_baselines.common import make_vec_env
from copy import copy
from gym import Env


def set_env(self, env):
    """
    Checks the validity of the environment, and if it is coherent, set it as the current environment.
    :param env: (Gym Environment) The environment for learning a policy
    """
    if env is None and self.env is None:
        if self.verbose >= 1:
            print("Loading a model without an environment, "
                  "this model cannot be trained until it has a valid environment.")
        return
    elif env is None:
        raise ValueError(
            "Error: trying to replace the current environment with None")

    # sanity checking the environment
    assert self.observation_space == env.observation_space, \
        "Error: the environment passed must have at least the same observation space as the model was trained on."
    assert self.action_space == Discrete(self.num_subpolicy), \
        "Error: the environment passed must have at least the same action space as the model was trained on."
    if self._requires_vec_env:
        assert isinstance(env, VecEnv), \
            "Error: the environment passed is not a vectorized environment, however {} requires it".format(
                self.__class__.__name__)
        assert not self.policy.recurrent or self.n_envs == env.num_envs, \
            "Error: the environment passed must have the same number of environments as the model was trained on." \
            "This is due to the Lstm policy not being capable of changing the number of environments."
        self.n_envs = env.num_envs
    else:
        # for models that dont want vectorized environment, check if they make sense and adapt them.
        # Otherwise tell the user about this issue
        if isinstance(env, VecEnv):
            if env.num_envs == 1:
                env = _UnvecWrapper(env)
                self._vectorize_action = True
            else:
                raise ValueError("Error: the model requires a non vectorized environment or a single vectorized "
                                 "environment.")
        else:
            self._vectorize_action = False

        self.n_envs = 1

    self.env = env

    # Invalidated by environment change.
    self.episode_reward = None
    self.ep_info_buf = None


class MasterModel(BaseRLModel):
    def __init__(self, policy, env, master_model_class, num_subpolicy=3, *args, **kwargs):
        super(MasterModel, self).__init__(policy, env, **kwargs)
        self.num_subpolicy = num_subpolicy
        master_model_class = copy(master_model_class)
        setattr(master_model_class, "set_env", set_env)
        if isinstance(env, VecEnv):
            temp_env = make_vec_env(
                [lambda: self.tempenv(env.observation_space, num_subpolicy)])
        else:
            temp_env = self.tempenv(env.observation_space, num_subpolicy)
        self.model = master_model_class(policy, temp_env)
        self.action_space = Discrete(self.num_subpolicy)

    @staticmethod
    def tempenv(observation_space, num_subpolicy):
        class TempEnv(object):
            def __init__(self):
                self.observation_space = observation_space
                self.action_space = Discrete(num_subpolicy)
        return TempEnv

    def set_env(self, env):
        """
        Checks the validity of the environment, and if it is coherent, set it as the current environment.
        :param env: (Gym Environment) The environment for learning a policy
        """
        if env is None and self.env is None:
            if self.verbose >= 1:
                print("Loading a model without an environment, "
                      "this model cannot be trained until it has a valid environment.")
            return
        elif env is None:
            raise ValueError(
                "Error: trying to replace the current environment with None")

        # sanity checking the environment
        assert self.observation_space == env.observation_space, \
            "Error: the environment passed must have at least the same observation space as the model was trained on."
        assert self.action_space == Discrete(self.num_subpolicy), \
            "Error: the environment passed must have at least the same action space as the model was trained on."
        if self._requires_vec_env:
            assert isinstance(env, VecEnv), \
                "Error: the environment passed is not a vectorized environment, however {} requires it".format(
                    self.__class__.__name__)
            assert not self.policy.recurrent or self.n_envs == env.num_envs, \
                "Error: the environment passed must have the same number of environments as the model was trained on." \
                "This is due to the Lstm policy not being capable of changing the number of environments."
            self.n_envs = env.num_envs
        else:
            # for models that dont want vectorized environment, check if they make sense and adapt them.
            # Otherwise tell the user about this issue
            if isinstance(env, VecEnv):
                if env.num_envs == 1:
                    env = _UnvecWrapper(env)
                    self._vectorize_action = True
                else:
                    raise ValueError("Error: the model requires a non vectorized environment or a single vectorized "
                                     "environment.")
            else:
                self._vectorize_action = False

            self.n_envs = 1

        self.env = env

        # Invalidated by environment change.
        self.episode_reward = None
        self.ep_info_buf = None

    def action_probability(self, observation, state=None, mask=None, actions=None, logp=False):
        return self.model.action_probability(observation, state, mask, actions, logp)

    def setup_model(self):
        pass

    def get_parameter_list(self):
        return self.model.get_parameter_list()

    def _get_pretrain_placeholders(self):
        return self.model._get_pretrain_placeholders()

    def learn(self, total_timesteps, callback=None, log_interval=100, tb_log_name="master",
              reset_num_timesteps=False):
        return self.model.learn(total_timesteps, callback=callback, log_interval=log_interval, tb_log_name=tb_log_name,
                                reset_num_timesteps=reset_num_timesteps)

    def predict(self, observation, state=None, mask=None, deterministic=False):
        return self.model.predict(observation, state, mask, deterministic)

    def save(self, save_path, cloudpickle=False):
        # TODO: how to save the master and sub models
        self.model.save(save_path, cloudpickle=cloudpickle)

    @classmethod
    def load(cls, load_path, env=None, custom_objects=None, **kwargs):
        # TODO: load the master model
        pass
