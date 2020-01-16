from stable_baselines.common.base_class import BaseRLModel, _UnvecWrapper
from copy import copy

class MLSH(BaseRLModel):
    def __init__(self, master_policy, env, master_model_class, subpolicy_timestep=200, num_subpolicy=3,
                 subpolicy_model_class=None,
                 subpolicy_model_policy=None, *args, **kwargs):
        super().__init__(policy=master_policy, env=env, verbose=kwargs.get('verbose', 0),
                         policy_base=None, requires_vec_env=True)

        if subpolicy_model_class is None:
            subpolicy_model_class = master_model_class

        if subpolicy_model_policy is None:
            subpolicy_model_policy = master_policy

        self.master_model_class = master_model_class
        self.sub_policy_model_class = subpolicy_model_class

        if env is not None:
            self.observation_space = env.observation_space
            self.action_space = env.action_space

        self.master_model = master_model_class(master_policy, env)

        self.subpolicies =[]
        for _ in range(num_subpolicy):
            self.subpolicies.append(self.sub_policy_model_class(subpolicy_model_policy, env))

        self.subpolicy_timestep = subpolicy_timestep

    def _check_obs(self, observation):
        if isinstance(observation, dict):
            if self.env is not None:
                if len(observation['observation'].shape) > 1:
                    observation = _UnvecWrapper.unvec_obs(observation)
                    return [self.env.convert_dict_to_obs(observation)]
                return self.env.convert_dict_to_obs(observation)
            else:
                raise ValueError("You must either pass an env to HER or wrap your env using HERGoalEnvWrapper")
        return observation

    def master_action_probability(self, observation, state=None, mask=None, actions=None, logp=False):
        return self.master_model.action_probability(self._check_obs(observation), state, mask, actions, logp)

    def subpolicy_action_probability(self, subpolicy_index, observation, state=None, mask=None, actions=None, logp=False):
        return self.subpolicies[subpolicy_index].action_probability\
            (self._check_obs(observation), state, mask, actions, logp)

    def action_probability(self, observation, state=None, mask=None, actions=None, logp=False):
        masters = self.master_model.action_probability()
