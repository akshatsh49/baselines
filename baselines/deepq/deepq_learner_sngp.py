from copy import deepcopy
import tensorflow as tf
from tensorflow import keras
import numpy as np

@tf.function
def huber_loss(x, delta=1.0):
    """Reference: https://en.wikipedia.org/wiki/Huber_loss"""
    return tf.where(
        tf.abs(x) < delta,
        tf.square(x) * 0.5,
        delta * (tf.abs(x) - 0.5 * delta)
    )

class DEEPQ_sngp(tf.Module):

    def __init__(self, q_func, observation_shape, num_actions, lr, grad_norm_clipping=None, gamma=1.0,
        double_q=True, param_noise=False, param_noise_filter_func=None):

      self.num_actions = num_actions
      self.gamma = gamma
      self.double_q = double_q
      self.param_noise = param_noise
      self.param_noise_filter_func = param_noise_filter_func
      self.grad_norm_clipping = grad_norm_clipping

      lr_schedule = keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=lr,
            decay_steps=int(5e4),
            end_learning_rate=5e-5,
      )

      self.optimizer = tf.keras.optimizers.legacy.Adam(
        decay = 1e-4, 
        learning_rate = lr_schedule,
        # learning_rate = lr,
      )

      with tf.name_scope('q_network'):
        self.q_network = q_func(observation_shape, num_actions)
      with tf.name_scope('target_q_network'):
        self.target_q_network = q_func(observation_shape, num_actions)
      
      self.eps = tf.Variable(0., name="eps")

    @tf.function
    def step(self, obs, stochastic=True, update_eps=-1, **step_args):
      if self.param_noise:
        raise ValueError('not supporting noise yet')
      else:
        q_values, cov_mat = self.q_network(obs)[0]

        deterministic_actions = tf.argmax(q_values, axis=1)
        batch_size = tf.shape(obs)[0]
        random_actions = tf.random.uniform(tf.stack([batch_size]), minval=0, maxval=self.num_actions, dtype=tf.int64)
        chose_random = tf.random.uniform(tf.stack([batch_size]), minval=0, maxval=1, dtype=tf.float32) < self.eps
        stochastic_actions = tf.where(chose_random, random_actions, deterministic_actions)

        if stochastic:
          output_actions = stochastic_actions
        else:
          output_actions = deterministic_actions

        if update_eps >= 0:
            self.eps.assign(update_eps)

        return output_actions, None, None, None

    @tf.function()
    def train(self, obs0, actions, rewards, obs1, dones, importance_weights):
      with tf.GradientTape() as tape:
        q_t, cov_mat = self.q_network(obs0)[0]
        q_t_selected = tf.reduce_sum(q_t * tf.one_hot(actions, self.num_actions, dtype=tf.float32), 1)

        q_tp1, cov_mat = self.target_q_network(obs1)[0]

        if self.double_q:
            q_tp1_using_online_net, cov_mat = self.q_network(obs1)[0]
            q_tp1_best_using_online_net = tf.argmax(q_tp1_using_online_net, 1)
            q_tp1_best = tf.reduce_sum(q_tp1 * tf.one_hot(q_tp1_best_using_online_net, self.num_actions, dtype=tf.float32), 1)
        else:
            q_tp1_best = tf.reduce_max(q_tp1, 1)

        dones = tf.cast(dones, q_tp1_best.dtype)
        q_tp1_best_masked = (1.0 - dones) * q_tp1_best

        q_t_selected_target = rewards + self.gamma * q_tp1_best_masked
        td_error = q_t_selected - tf.stop_gradient(q_t_selected_target)
        errors = huber_loss(td_error)
        weighted_error = tf.reduce_mean(importance_weights * errors)

      grads = tape.gradient(weighted_error, self.q_network.trainable_variables)
      if self.grad_norm_clipping:
        clipped_grads = []
        for grad in grads:
          clipped_grads.append(tf.clip_by_norm(grad, self.grad_norm_clipping))
        grads = clipped_grads
      grads_and_vars = zip(grads, self.q_network.trainable_variables)
      self.optimizer.apply_gradients(grads_and_vars)

      return td_error, q_t_selected

    @tf.function(autograph=False)
    def update_target(self):
      q_vars = self.q_network.trainable_variables
      target_q_vars = self.target_q_network.trainable_variables
      for var, var_target in zip(q_vars, target_q_vars):
        var_target.assign(var)

    @tf.function
    def uncertainty_estimate(self, obs):
      q, covmat = self.q_network(obs, training = False)[0]
      return covmat
    
    def get_network_weights(self):
        return {
            'q_net' : self.q_network.trainable_variables, 
            'target_net' : self.target_q_network.trainable_variables, 
        }



