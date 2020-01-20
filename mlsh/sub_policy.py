import time
import numpy as np

import tensorflow as tf
from collections import deque
from stable_baselines.trpo_mpi.trpo_mpi import TRPO
from mpi4py import MPI
from stable_baselines.ppo2.ppo2 import get_schedule_fn
from stable_baselines import logger
from stable_baselines.common import explained_variance, dataset
from stable_baselines.common.cg import conjugate_gradient
from stable_baselines.trpo_mpi.utils import  add_vtarg_and_adv, flatten_lists
from stable_baselines.common.vec_env import VecEnv


def traj_segment_generator(policy, env, horizon, callback, reward_giver=None, gail=False):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
    :param policy: (MLPPolicy) the policy
    :param env: (Gym Environment) the environment
    :param horizon: (int) the number of timesteps to run per batch
    :param reward_giver: (TransitionClassifier) the reward predicter from obsevation and action
    :param gail: (bool) Whether we are using this generator for standard trpo or with gail
    :return: (dict) generator that returns a dict with the following keys:
        - observations: (np.ndarray) observations
        - rewards: (numpy float) rewards (if gail is used it is the predicted reward)
        - true_rewards: (numpy float) if gail is used it is the original reward
        - vpred: (numpy float) action logits
        - dones: (numpy bool) dones (is end of episode, used for logging)
        - episode_starts: (numpy bool)
            True if first timestep of an episode, used for GAE
        - actions: (np.ndarray) actions
        - nextvpred: (numpy float) next action logits
        - ep_rets: (float) cumulated current episode reward
        - ep_lens: (int) the length of the current episode
        - ep_true_rets: (float) the real environment reward
    """
    # Check when using GAIL
    assert not (gail and reward_giver is None), "You must pass a reward giver when using GAIL"

    # Initialize state variables
    step = 0
    action = env.action_space.sample()  # not used, just so we have the datatype
    observation = env.reset()

    cur_ep_ret = 0  # return in current episode
    current_it_len = 0  # len of current iteration
    current_ep_len = 0 # len of current episode
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
    end = False
    while True:
        action, vpred, states, _ = policy.step(observation.reshape(-1, *observation.shape), states, done)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if step > 0 and step % horizon == 0:
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
                    "total_timestep": current_it_len
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
        if isinstance(env.action_space, gym.spaces.Box):
            clipped_action = np.clip(action, env.action_space.low, env.action_space.high)

        if gail:
            reward = reward_giver.get_reward(observation, clipped_action[0])
            observation, true_reward, done, info = env.step(clipped_action[0])
        else:
            observation, reward, done, info = env.step(clipped_action[0])
            true_reward = reward
        rewards[i] = reward
        true_rewards[i] = true_reward
        dones[i] = done
        episode_start = done

        cur_ep_ret += reward
        cur_ep_true_ret += true_reward
        current_it_len += 1
        current_ep_len += 1
        if done:
            print("transition done", done)
            # Retrieve unnormalized reward if using Monitor wrapper
            maybe_ep_info = info.get('episode')
            if maybe_ep_info is not None:
                if not gail:
                    cur_ep_ret = maybe_ep_info['r']
                cur_ep_true_ret = maybe_ep_info['r']

            ep_rets.append(cur_ep_ret)
            ep_true_rets.append(cur_ep_true_ret)
            ep_lens.append(current_ep_len)
            cur_ep_ret = 0
            cur_ep_true_ret = 0
            current_ep_len = 0
            if not end:
                end = True
                callback["done"] = True
                callback["episode_len"] = current_ep_len
            if not isinstance(env, VecEnv):
                observation = env.reset()
        step += 1



class MLSHSubpolicy(TRPO):
    def __init__(self, policy, env, gamma=0.99, timesteps_per_batch=1024, max_kl=0.01, cg_iters=10, lam=0.98,
                 entcoeff=0.0, cg_damping=1e-2, vf_stepsize=3e-4, vf_iters=3, verbose=0, tensorboard_log=None,
                 _init_setup_model=True, policy_kwargs=None, full_tensorboard_log=False,
                 seed=None, n_cpu_tf_sess=1):
        super(MLSHSubpolicy, self).__init__(policy=policy, env=env, verbose=verbose, timesteps_per_batch=timesteps_per_batch,
                                   max_kl=max_kl, lam=lam, entcoeff=entcoeff, cg_damping=cg_damping,
                                   vf_stepsize=vf_stepsize, vf_iters=vf_iters, tensorboard_log=tensorboard_log,
                                   _init_setup_model=_init_setup_model, policy_kwargs=policy_kwargs,
                                   full_tensorboard_log=full_tensorboard_log, seed=seed, n_cpu_tf_sess=n_cpu_tf_sess,
                                            gamma=gamma, cg_iters=cg_iters)
        self.setup_learn_flag = False
        self.async_scheduler = None
        self.learning_rate_scheduler = None
        self.cliprange_scheduler = None
        self.cliprange_vf_scheduler = None
        self.timesteps_so_far = 0
        self.iters_so_far = 0
        self.master_callback = None

    def reset_learning_setting(self, env):
        self.env = env
        self.learning_rate_scheduler = get_schedule_fn(self.vf_stepsize)
        self.timesteps_so_far = 0
        self.iters_so_far = 0

    def learning_setting(self, env, n_steps):
        self.env = env
        self.n_batch = n_steps

    def _setup_learn(self):
        self._setup_learn_flag = True
        return super(MLSHSubpolicy, self)._setup_learn()

    def learn_async(self, total_timesteps, env, writer=None):

        with self.sess.as_default():
            self.master_callback = {"rewards":0, "obs": None, "done": False, "episode_len":0}
            seg_gen = traj_segment_generator(self.policy_pi, env, total_timesteps, callback=self.master_callback,
                                             reward_giver=self.reward_giver, gail=False)

            len_buffer = deque(maxlen=40)  # rolling buffer for episode lengths
            reward_buffer = deque(maxlen=40)  # rolling buffer for episode rewards
            true_reward_buffer = None
            timesteps_so_far = self.timesteps_so_far
            iters_so_far = self.iters_so_far

            gamma = self.gamma
            end = False
            master_reward = 0
            master_episode_len = 0
            while True:
                if total_timesteps and timesteps_so_far >= total_timesteps:
                    break

                logger.log("********** Iteration %i ************" % iters_so_far)

                def fisher_vector_product(vec):
                    return self.allmean(self.compute_fvp(vec, *fvpargs, sess=self.sess)) + self.cg_damping * vec

                # ------------------ Update G ------------------
                logger.log("Optimizing Policy...")
                # g_step = 1 when not using GAIL
                mean_losses = None
                vpredbefore = None
                tdlamret = None
                observation = None
                action = None
                seg = None
                done = 0
                with self.timed("sampling"):
                    seg = seg_gen.__next__()
                add_vtarg_and_adv(seg, self.gamma, self.lam)

                observation, action = seg["observations"], seg["actions"]

                atarg, tdlamret = seg["adv"], seg["tdlamret"]
                endpoint = self.endpoint(seg["dones"])
                print("dones", seg["dones"])
                if endpoint != len(seg["dones"]) and not end:
                    self.master_callback["done"] = True

                    master_reward += np.sum(seg["rewards"][:endpoint])
                    end = True
                    master_episode_len += endpoint
                if not end:
                    master_reward += np.sum(seg["rewards"])
                    master_episode_len += endpoint
                done = np.clip(done, a_max=1, a_min=0)
                vpredbefore = seg["vpred"]  # predicted value function before update
                atarg = (atarg - atarg.mean()) / atarg.std()  # standardized advantage function estimate

                args = seg["observations"], seg["observations"], seg["actions"], atarg
                # Subsampling: see p40-42 of John Schulman thesis

                # http://joschu.net/docs/thesis.pdf
                fvpargs = [arr[::5] for arr in args]

                self.assign_old_eq_new(sess=self.sess)

                with self.timed("computegrad"):
                    steps = self.num_timesteps + seg["total_timestep"]
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata() if self.full_tensorboard_log else None
                    # run loss backprop with summary, and save the metadata (memory, compute time, ...)
                    if writer is not None:
                        summary, grad, *lossbefore = self.compute_lossandgrad(*args, tdlamret, sess=self.sess,
                                                                              options=run_options,
                                                                              run_metadata=run_metadata)
                        if self.full_tensorboard_log:
                            writer.add_run_metadata(run_metadata, 'step%d' % steps)
                        writer.add_summary(summary, steps)
                    else:
                        _, grad, *lossbefore = self.compute_lossandgrad(*args, tdlamret, sess=self.sess,
                                                                        options=run_options,
                                                                        run_metadata=run_metadata)

                lossbefore = self.allmean(np.array(lossbefore))
                grad = self.allmean(grad)
                if np.allclose(grad, 0):
                    logger.log("Got zero gradient. not updating")
                else:
                    with self.timed("conjugate_gradient"):
                        stepdir = conjugate_gradient(fisher_vector_product, grad, cg_iters=self.cg_iters,
                                                     verbose=self.rank == 0 and self.verbose >= 1)
                    assert np.isfinite(stepdir).all()
                    shs = .5 * stepdir.dot(fisher_vector_product(stepdir))
                    # abs(shs) to avoid taking square root of negative values
                    lagrange_multiplier = np.sqrt(abs(shs) / self.max_kl)
                    # logger.log("lagrange multiplier:", lm, "gnorm:", np.linalg.norm(g))
                    fullstep = stepdir / lagrange_multiplier
                    expectedimprove = grad.dot(fullstep)
                    surrbefore = lossbefore[0]
                    stepsize = 1.0
                    thbefore = self.get_flat()
                    for _ in range(10):
                        thnew = thbefore + fullstep * stepsize
                        self.set_from_flat(thnew)
                        mean_losses = surr, kl_loss, *_ = self.allmean(
                            np.array(self.compute_losses(*args, sess=self.sess)))
                        improve = surr - surrbefore
                        logger.log("Expected: %.3f Actual: %.3f" % (expectedimprove, improve))
                        if not np.isfinite(mean_losses).all():
                            logger.log("Got non-finite value of losses -- bad!")
                        elif kl_loss > self.max_kl * 1.5:
                            logger.log("violated KL constraint. shrinking step.")
                        elif improve < 0:
                            logger.log("surrogate didn't improve. shrinking step.")
                        else:
                            logger.log("Stepsize OK!")
                            break
                        stepsize *= .5
                    else:
                        logger.log("couldn't compute a good step")
                        self.set_from_flat(thbefore)
                    if self.nworkers > 1 and iters_so_far % 20 == 0:
                        # list of tuples
                        paramsums = MPI.COMM_WORLD.allgather((thnew.sum(), self.vfadam.getflat().sum()))
                        assert all(np.allclose(ps, paramsums[0]) for ps in paramsums[1:])

                    for (loss_name, loss_val) in zip(self.loss_names, mean_losses):
                        logger.record_tabular(loss_name, loss_val)
                with self.timed("vf"):
                    for _ in range(self.vf_iters):
                        # NOTE: for recurrent policies, use shuffle=False?
                        for (mbob, mbret) in dataset.iterbatches((seg["observations"], seg["tdlamret"]),
                                                                 include_final_partial_batch=False,
                                                                 batch_size=32,
                                                                 shuffle=True):
                            grad = self.allmean(self.compute_vflossandgrad(mbob, mbob, mbret, sess=self.sess))
                            self.vfadam.update(grad, self.vf_stepsize)

                logger.record_tabular("explained_variance_tdlam_before",
                                      explained_variance(vpredbefore, tdlamret))

                # lr: lengths and rewards
                lr_local = (seg["ep_lens"], seg["ep_rets"])  # local values
                list_lr_pairs = MPI.COMM_WORLD.allgather(lr_local)  # list of tuples
                lens, rews = map(flatten_lists, zip(*list_lr_pairs))
                len_buffer.extend(lens)
                reward_buffer.extend(rews)

                if len(len_buffer) > 0:
                    logger.record_tabular("EpLenMean", np.mean(len_buffer))
                    logger.record_tabular("EpRewMean", np.mean(reward_buffer))

                logger.record_tabular("EpThisIter", len(lens))
                current_it_timesteps = MPI.COMM_WORLD.allreduce(seg["total_timestep"])
                timesteps_so_far += current_it_timesteps
                self.num_timesteps += current_it_timesteps
                iters_so_far += 1
                last_obs = seg["observations"][-1]
                if self.verbose >= 1 and self.rank == 0:
                    logger.dump_tabular()
        self.master_callback["obs"] = last_obs
        self.master_callback["rewards"] = master_reward
        self.master_callback["episode_len"] = master_episode_len
        return self

    @staticmethod
    def endpoint(dones):
        end_point = np.where(dones == 1)[0]
        # there were no end point
        if len(end_point) == 0:
            return len(dones)
        # return smallest end pint
        else:
            return end_point[0]


import gym
from stable_baselines.common.policies import MlpPolicy
if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    model = MLSHSubpolicy(env=env, policy=MlpPolicy, verbose=1)
    model.reset_learning_setting(env)
    print("regular learning")
    model.learn(total_timesteps=1000)
    print("async learning ")
    for i in range(100):
        model.learn_async(total_timesteps=100, env=env)
        print(model.master_callback)
    print(model.master_callback)

