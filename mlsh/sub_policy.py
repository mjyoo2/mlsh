import numpy as np
import tensorflow as tf

from stable_baselines.ppo2.ppo2 import get_schedule_fn
from stable_baselines.common.math_util import unscale_action
from stable_baselines import SAC
from stable_baselines import DQN
from stable_baselines.deepq.dqn import PrioritizedReplayBuffer, ReplayBuffer
from stable_baselines.common.schedules import LinearSchedule


class MLSHSubpolicySAC(SAC):
    def __init__(self, *sac_args, **sac_kwargs):
        super(MLSHSubpolicySAC, self).__init__(*sac_args, **sac_kwargs)
        self.async_lr_scheduler = get_schedule_fn(self.learning_rate)
        self.async_step = 1
        self.current_lr = self.async_lr_scheduler(1)

    def rollout_async(self, obs, env):
        """
        Method conducts "predict and insert into the replay buffer"
        :param obs: observation, and environment
        :return: next observation, reward, done, info
        """

        action = self.policy_tf.step(obs[None], deterministic=False).flatten()
        # Add noise to the action (improve exploration,
        # not needed in general)
        if self.action_noise is not None:
            action = np.clip(action + self.action_noise(), -1, 1)
        # inferred actions need to be transformed to environment action_space before stepping
        unscaled_action = unscale_action(env.action_space, action)
        new_obs, reward, done, info = self.env.step(unscaled_action)
        self.replay_buffer.add(obs, action, reward, new_obs, float(done))
        return new_obs, reward, done, info

    def learn_async(self, writer=None, frac=1):

        if self.async_step % self.train_freq == 0:
            # Update policy, critics and target networks
            for grad_step in range(self.gradient_steps):
                # Break if the warmup phase is not over
                # or if there are not enough samples in the replay buffer
                if not self.replay_buffer.can_sample(self.batch_size) \
                        or self.async_step < self.learning_starts:
                    break
                self.async_step+= 1
                current_lr = self.async_lr_scheduler(frac)
                # Update policy and critics (q functions)
                self._train_step(self.async_step, writer, current_lr)
                # Update target network
                if (self.async_step + grad_step) % self.target_update_interval == 0:
                    # Update target network
                    self.sess.run(self.target_update_op)
        self.async_step += 1
        return self

    def clear_replay_buffer(self):
        del self.replay_buffer
        self.replay_buffer = ReplayBuffer(self.buffer_size)

    def setup_learn(self):
        return super()._setup_learn()


class MLSHSubpolicyDQN(DQN):
    def __init__(self, exploration_steps=10000, replay_wrapper=None, *args, **kwargs):
        super(MLSHSubpolicyDQN, self).__init__(*args, **kwargs)
        self.async_lr_scheduler = get_schedule_fn(self.learning_rate)
        self.async_step = 1
        self.current_lr = self.async_lr_scheduler(1)
        self.name = "normal"
        self.exploration = LinearSchedule(schedule_timesteps=int(self.exploration_fraction * exploration_steps),
                                              initial_p=self.exploration_initial_eps,
                                              final_p=self.exploration_final_eps)
        self.replay_wrapper = replay_wrapper

    def rollout_async(self, obs, env):
        update_eps = self.exploration.value(self.async_step)
        with self.sess.as_default():
            action = self.act(np.array(obs)[None], update_eps=update_eps, stochastic=False)[0]

        new_obs, rew, done, info = env.step(action)

        # Store transition in the replay buffer.
        self.replay_buffer.add(np.copy(obs), action, rew, np.copy(new_obs), float(done))
        return new_obs, rew, done, info

    def learn_async(self, writer=None, frac=1):
        # Do not train if the warmup phase is not over
        # or if there are not enough samples in the replay buffer
        can_sample = self.replay_buffer.can_sample(self.batch_size)
        if can_sample and self.async_step > self.learning_starts \
                and self.async_step % self.train_freq == 0:
            # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
            # pytype:disable=bad-unpacking
            if self.prioritized_replay:
                assert self.beta_schedule is not None, \
                    "BUG: should be LinearSchedule when self.prioritized_replay True"
                experience = self.replay_buffer.sample(self.batch_size,
                                                       beta=self.beta_schedule.value(self.async_step))
                (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
            else:
                obses_t, actions, rewards, obses_tp1, dones = self.replay_buffer.sample(self.batch_size)
                weights, batch_idxes = np.ones_like(rewards), None
            # pytype:enable=bad-unpacking

            if writer is not None:
                # run loss backprop with summary, but once every 100 steps save the metadata
                # (memory, compute time, ...)
                if (1 + self.async_step) % 100 == 0:
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    summary, td_errors = self._train_step(obses_t, actions, rewards, obses_tp1, obses_tp1,
                                                          dones, weights, sess=self.sess, options=run_options,
                                                          run_metadata=run_metadata)
                    writer.add_run_metadata(run_metadata, 'step%d' % self.async_step)
                else:
                    summary, td_errors = self._train_step(obses_t, actions, rewards, obses_tp1, obses_tp1,
                                                          dones, weights, sess=self.sess)
                writer.add_summary(summary, self.async_step)
            else:
                _, td_errors = self._train_step(obses_t, actions, rewards, obses_tp1, obses_tp1, dones, weights,
                                                sess=self.sess)

            if self.prioritized_replay:
                new_priorities = np.abs(td_errors) + self.prioritized_replay_eps
                assert isinstance(self.replay_buffer, PrioritizedReplayBuffer)
                self.replay_buffer.update_priorities(batch_idxes, new_priorities)

            if can_sample and self.num_timesteps > self.learning_starts and \
                    self.async_step % self.target_network_update_freq == 0:
                # Update target network periodically.
                self.update_target(sess=self.sess)
        self.async_step += 1
        return self

    def setup_learn(self):
        self._setup_learn()
        # Create the replay buffer
        if self.prioritized_replay:
            raise NotImplementedError("PER has not implemented yet")
        else:
            self.replay_buffer = ReplayBuffer(self.buffer_size)
            self.beta_schedule = None

        if self.replay_wrapper is not None:
            assert not self.prioritized_replay, "Prioritized replay buffer is not supported by HER"
            self.replay_buffer = self.replay_wrapper(self.replay_buffer)
