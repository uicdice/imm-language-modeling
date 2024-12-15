r"""
This is the second of the three repositories accompanying the paper
    Induced Model Matching: Restricted Models Help Train Full-Featured Models (NeurIPS 2024)

This repository demonstrates Induced Model Matching in the training of an LSTM RNN

Other implementations are contained in the following repositories:
    IMM in Logistic Regression: https://github.com/uicdice/imm-logistic-regression
    IMM in learning MDPs (REINFORCE): https://github.com/uicdice/imm-reinforce

This codebase is based on the codebase of our main baseline
    Data Noising as Smoothing in Neural Network Language Models (ICLR 2017)
    https://github.com/stanfordmlgroup/nlm-noising

which in turn is based on a TensorFlow Official Models adaptation of
    https://github.com/wojzaremba/lstm


This codebase is covered by

Copyright 2021-2024 Usama Muneeb and Mesrob Ohannessian

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import copy
import logging
import os
import time
from collections import namedtuple
from os.path import join as pjoin

import numpy as np
import tensorflow.compat.v1 as tf

from cfg import PTB_DATA_PATHS
from loader import TextLoader

logging.basicConfig(level=logging.INFO)

flags = tf.flags

# Settings
flags.DEFINE_integer("hidden_dim", 1500, "hidden dimension")
flags.DEFINE_integer("layers", 2, "number of hidden layers")
flags.DEFINE_integer("unroll", 35, "number of time steps to unroll for BPTT")
flags.DEFINE_integer("batch_size", 20, "batch size")
flags.DEFINE_float("init_scale", 0.1, "scale for random initialization")
flags.DEFINE_float("learning_rate", 1.0, "initial learning rate")
flags.DEFINE_float("lambda_param", 0.2, "IMM regularization coefficient")
flags.DEFINE_float("start_imm_at", 90.0, "Perplexity below which IMM regularization will start")
flags.DEFINE_boolean("imm_started", False,
  "Whether to start IMM immediately or after perplexity gets below start_imm_at")
flags.DEFINE_integer("unrolls_to_use", 35,
  "number of time steps to use for custom loss (starting from end)")
flags.DEFINE_integer("long_hist_samples", 10,
  "number of long histories to randomly sample for second loss")
flags.DEFINE_float("learning_rate_decay", 0.5, "amount to decrease learning rate")
flags.DEFINE_float("decay_threshold", 0.0,
  "decrease learning rate if validation cost difference less than this value")
flags.DEFINE_integer("max_decays", 20, "stop decreasing learning rate after this many times")
flags.DEFINE_float("drop_prob", 0.0, "probability of dropping units")
flags.DEFINE_integer("max_epochs", 400, "maximum number of epochs to train")
flags.DEFINE_float("clip_norm", 5.0, "value at which to clip gradients")
flags.DEFINE_string("optimizer", "sgd", "optimizer")
flags.DEFINE_string("run_dir", "sandbox", "directory to store experiment outputs")
flags.DEFINE_string("token_type", "word", "use word or character tokens")
flags.DEFINE_integer("sample_every", 5, "Sample every nth batch")
flags.DEFINE_integer("simultaneous_samples", 1,
  "Numbers of random samples to be evaluated simultaneously on GPU (use 5 for Nvidia V100)")
flags.DEFINE_string("restore_checkpoint", None, "checkpoint file to restore model parameters from")
flags.DEFINE_integer("seed", 123, "random seed to use")
flags.DEFINE_integer("prints_per_epoch", 10, "Number of updates per epoch.")
flags.DEFINE_integer("steps_per_summary", 10, "how many steps between writing summaries")
flags.DEFINE_boolean("final", False, "final evaluation (run on test after picked best model)")
flags.DEFINE_string("dataset", "ptb", "ptb or text8")

FLAGS = flags.FLAGS

# Getting stale file handle errors
def log_info(s):
  try:
    logging.info(s)
  except IOError:
    time.sleep(60)



# This is a rewrite of the deprecated `sequence_loss_by_example` (used by Xie et al paper) for
# compatibility with TensorFlow 2.x
# This assumes one hot targets and takes labels as targets.
def defaultLoss(logits, languageModel, flags, vocab_size):
    y = languageModel._y

    # seq_loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
    #     [tf.reshape(logits, [-1, vocab_size])],
    #     [tf.reshape(y, [-1])],
    #     [tf.ones([flags.batch_size * flags.unroll])]
    # )
    # return tf.reduce_sum(seq_loss) / flags.batch_size

    exp_logits = tf.math.exp(logits)
    denominators = tf.math.reduce_sum(exp_logits, axis=1)
    y_slice = tf.reshape(y, [flags.batch_size * flags.unroll])
    numerators = tf.gather(
      tf.reshape(exp_logits, [-1]),
      tf.range(flags.batch_size * flags.unroll) * vocab_size + y_slice
    )
    seq_loss = -tf.math.log(tf.divide(numerators, denominators))
    return tf.reduce_sum(seq_loss) / flags.batch_size

def IMM_Loss(logits, languageModel, flags, vocab_size):
    # get the Kneser-Ney target (i.e. p(y)) fed to the graph.
    p_y = languageModel._y_imm

    # get dot product of p(y) and logits
    # both are of dimensions (batch_size * unroll, vocab_size)
    # element wise multiplication followed by summation along `vocab_size` axis
    # will end up with tensor of shape (batch_size * unroll, )
    numerators = tf.math.reduce_sum(
      tf.multiply(logits, p_y),
      axis=-1 # sum up along `vocab_size` dimension
    )

    # get the sum-exp of logits (for log-sum-exp term)
    # we write the three steps separately as this easily generalizes
    # the version of this loss with cross talk
    exp_logits = tf.math.exp(logits)
    sumexp = tf.math.reduce_sum(exp_logits, axis=-1)

    # -p(y) dot logits + log-sum-exp of logits
    cross_entropy = - numerators + tf.math.log(sumexp)

    # reduce and normalize
    cross_entropy = tf.reduce_sum(cross_entropy) / flags.batch_size

    return cross_entropy

def IMM_Loss_sampling(logits, languageModel, flags, vocab_size, crosstalk):
    # get the Kneser-Ney target (i.e. p(y)) fed to the graph.
    p_y = languageModel._y_imm_sampling

    # get dot product of doctored p(y) and logits (doctored using `crosstalk`)
    # both are of dimensions (batch_size * unroll, vocab_size)
    # element wise multiplication followed by summation along `vocab_size` axis
    # will end up with tensor of shape (batch_size * unroll, )
    numerators = tf.math.reduce_sum(
      tf.multiply(logits, tf.multiply(p_y, crosstalk)),
      axis=-1 # sum up along `vocab_size` dimension
    )

    # get the sum-exp of logits (for log-sum-exp term)
    # we write the three steps separately as this easily generalizes
    # the version of this loss with cross talk
    exp_logits = tf.math.exp(logits)
    sumexp = tf.math.reduce_sum(exp_logits, axis=-1)

    # q(y)
    # this is used to compute crosstalk which will be fed back into the graph
    # to enable computation of doctored cross entropy
    q_y = exp_logits / tf.expand_dims(sumexp, -1)

    # -p(y) dot logits + log-sum-exp of logits
    doctored_cross_entropy = - numerators + tf.multiply(
      tf.math.log(sumexp),
      tf.reduce_sum(tf.multiply(p_y, crosstalk), -1)
    )

    # reduce and normalize
    doctored_cross_entropy = tf.reduce_sum(doctored_cross_entropy) / flags.batch_size

    return doctored_cross_entropy, q_y


def netQ(languageModel, inputSource, flags, vocab_size,
         is_training, forceReuse=False, fixed_unroll=True):
  batch_size = flags.batch_size
  unroll = flags.unroll

  if forceReuse:
    tf.get_variable_scope().reuse_variables()

  lstm_cells = list()
  for k in range(flags.layers):
    lstm_cell = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(
      flags.hidden_dim, forget_bias=1.0, state_is_tuple=True,
      reuse=tf.get_variable_scope().reuse) # this gets the reuse attribute.
    if is_training and fixed_unroll and flags.drop_prob > 0:
      lstm_cell = tf.compat.v1.nn.rnn_cell.DropoutWrapper(
          lstm_cell , output_keep_prob=1.0-flags.drop_prob)
    lstm_cells.append(lstm_cell)
  cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(lstm_cells, state_is_tuple=True)

  # `session.run` can treat state as a `placeholder` but not `session.partial_run`
  # so we use a separate object for the initializer and placeholders for providing initial state
  languageModel._initializer = cell.zero_state(batch_size, tf.float32)

  with tf.device("/cpu:0"):
    languageModel.embeddings = tf.get_variable("embeddings", [vocab_size, flags.hidden_dim])
    inputs = tf.nn.embedding_lookup(languageModel.embeddings, inputSource)
  if is_training and fixed_unroll and flags.drop_prob > 0:
    inputs = tf.nn.dropout(inputs, 1.0 - flags.drop_prob)

  # These options (fixed unroll or dynamic_rnn) should give same results but
  # using fixed here since faster

  state = (
    tf.nn.rnn_cell.LSTMStateTuple(languageModel._initial_state_0_c,
                                  languageModel._initial_state_0_h),
    tf.nn.rnn_cell.LSTMStateTuple(languageModel._initial_state_1_c,
                                  languageModel._initial_state_1_h)
  )
  if fixed_unroll:
    outputs = []
    with tf.variable_scope("RNN"):
      for time_step in range(unroll):
        if time_step > 0:
          # we should reuse the same cell for `time_step` > 0 instead of creating new ones
          tf.get_variable_scope().reuse_variables()
        (cell_output, state) = cell(inputs[:, time_step, :], state)
        outputs.append(cell_output)
    outputs = tf.reshape(tf.concat(outputs, 1), [-1, flags.hidden_dim])
  else:
    outputs, state = tf.nn.dynamic_rnn(
      cell,
      inputs,
      sequence_length=tf.repeat(languageModel._lookingAt+tf.constant(1), repeats=flags.batch_size),
      initial_state=state, dtype=tf.float32, time_major=False, scope="RNN"
    )
    outputs = outputs[:,languageModel._lookingAt,:]

  softmax_w = tf.get_variable("softmax_w", [flags.hidden_dim, vocab_size])
  softmax_b = tf.get_variable("softmax_b", [vocab_size])
  logits = tf.matmul(outputs, softmax_w) + softmax_b

  # NOTE: the `logits` returned are NOT softmaxed
  # Non-softmaxed outputs can be used by the loss functions directly to produce more efficient code

  return state, logits






class LanguageModel(object):

  def __init__(self, flags, vocab_size, is_training=True):
    batch_size = flags.batch_size
    unroll = flags.unroll

    # self._len = tf.placeholder(tf.int32, [None, ])

    # For partial run, state cannot be provided as a tuple, so we split into components instead
    self._initial_state_0_c = tf.placeholder(tf.float32, [None, flags.hidden_dim])
    self._initial_state_0_h = tf.placeholder(tf.float32, [None, flags.hidden_dim])
    self._initial_state_1_c = tf.placeholder(tf.float32, [None, flags.hidden_dim])
    self._initial_state_1_h = tf.placeholder(tf.float32, [None, flags.hidden_dim])

    # these placeholders correspond to the main (cross entropy loss)
    self._x = tf.placeholder(tf.int32, [batch_size, unroll])
    self._y = tf.placeholder(tf.int32, [batch_size, unroll])

    # state will only be updated by the feedforward of the main loss
    self._final_state, logits = netQ(self, self._x, flags, vocab_size, is_training)

    # at this stage, all trainable variables have been created
    self.tvars = tf.trainable_variables()
    shapes = [tvar.get_shape() for tvar in self.tvars]
    log_info("# params: %d" % np.sum([np.prod(s) for s in shapes]))

    self.loss = defaultLoss(logits, self, flags, vocab_size)
    self.grad = tf.gradients(self.loss, self.tvars)


    if flags.clip_norm is not None:
      self.grad, grads_norm = tf.clip_by_global_norm(self.grad, flags.clip_norm)
    else:
      grads_norm = tf.global_norm(self.grad)

    if is_training:
      # placeholders for the single sample IMM component
      # these ones are used by the single sample IMM
      self._x_imm = tf.placeholder(tf.int32, [batch_size, unroll])
      self._y_imm = tf.placeholder(tf.float32, [batch_size * unroll, vocab_size])

      _, logits_imm = netQ(self, self._x_imm, flags, vocab_size, is_training, True)
      self.loss_imm = FLAGS.lambda_param * IMM_Loss(logits_imm, self, flags, vocab_size)
      self.grad_imm = tf.gradients(self.loss_imm, self.tvars)

      # placeholders for k-sample IMM component
      # used every `FLAGS.sample_every` iteration
      self._x_imm_sampling = []
      self._y_imm_sampling = tf.placeholder(tf.float32, [batch_size, vocab_size])
      self._lookingAt = tf.placeholder(tf.int32)

      self.q_y_out = [] # extracted during forward pass (to compute crosstalk)
      self.crosstalk_in = [] # computed using `q_y_out` and fed back during backward pass
      self.grads_imm_per_sample = []

      # create a separate forward graph for each of k random samples
      # allows us to conveniently continue execution after providing crosstalks
      for j in range(flags.long_hist_samples):

        # NOTE: since we random sample separately for each unroll position
        # this means the input should be truncated from the beginning
        # the amount of truncation depends on the unroll position we are looking at
        # more details in training loop
        sample_input = tf.placeholder(tf.int32, [batch_size, unroll])

        # additionally, k-sample IMM uses `tf.nn.dynamic_rnn` so we do not end up computing
        # outputs for unroll positions beyond the one we are looking at
        _, logits_sample = netQ(self, sample_input, flags, vocab_size, is_training, True, False)

        crosstalk = tf.placeholder(tf.float32, [batch_size, vocab_size])

        # `sample_doctored_cross_entropy` cannot be computed during forward pass as `crosstalk`
        # tensor is not yet available/fed. It will only be computed during backward pass after
        # `crosstalk` tensor is provided. Furthermore, it's a doctored loss for sequentializing
        # gradient computation. Therefore, its value should not be considered for printing.

        sample_doctored_cross_entropy, q_y = \
          IMM_Loss_sampling(logits_sample, self, flags, vocab_size, crosstalk)
        sample_doctored_cross_entropy = FLAGS.lambda_param * sample_doctored_cross_entropy

        # save input node
        self._x_imm_sampling.append(sample_input)

        # save (intermediate) output node
        self.q_y_out.append(q_y)
        # save intermediate input node
        self.crosstalk_in.append(crosstalk)

        # While the gradients are also doctored, the sum of all sample gradients will be the exact
        # IMM gradient (as we also discuss in the Appendix of our paper).
        sample_doctored_grad = tf.gradients(
          sample_doctored_cross_entropy,
          tf.trainable_variables()
        )

        self.grads_imm_per_sample.append(sample_doctored_grad)



      self.lr = tf.Variable(0.0, trainable=False)
      if flags.optimizer == "sgd":
        optimizer = tf.train.GradientDescentOptimizer(self.lr)
      elif flags.optimizer == "adam":
        optimizer = tf.train.AdamOptimizer(self.lr)
      else:
        assert False, "Optimizer must be SGD or Adam"


      # this (simple) variant used before IMM regularization steps in
      self._train_op = optimizer.apply_gradients(zip(self.grad, self.tvars))

      # placeholders to feed back gradients accumulated on CPU
      self.gradientPlaceholders = {}
      self.gradientPlaceholders[self.tvars[0].name+"_indices"] = tf.placeholder(tf.int32)
      self.gradientPlaceholders[self.tvars[0].name+"_values"] = tf.placeholder(tf.float32)
      embeddingGradient = tf.IndexedSlices(
        self.gradientPlaceholders[self.tvars[0].name+"_values"],
        self.gradientPlaceholders[self.tvars[0].name+"_indices"],
        dense_shape=[vocab_size, flags.hidden_dim]
      )
      for tvar in self.tvars[1:]:
        self.gradientPlaceholders[tvar.name] = tf.placeholder(tf.float32)

      total_grads = [embeddingGradient] + [
        self.gradientPlaceholders[tvar.name] for tvar in self.tvars[1:]
      ]

      # this was needed only when trying AdamOptimizer
      # tf.get_variable_scope()._reuse = False
      self._train_op_imm = optimizer.apply_gradients(zip(total_grads, self.tvars))

      with tf.name_scope("summaries"):
        tf.summary.scalar("loss", self.loss / unroll)
        tf.summary.scalar("learning_rate", self.lr)
        tf.summary.scalar("grads_norm", grads_norm)

      self.infer_only = False
      log_info("Created trainable model object")
    else:
      self.infer_only = True
      log_info("Created non-trainable model object")


  def set_lr(self, session, lr_value):
    session.run(tf.assign(self.lr, lr_value))


def combine_gradients(gradient_sets, tvars):
  num_grad_sets = len(gradient_sets)
  combined_grads = [0] * len(tvars) # number of layers after embedding

  EmbeddingGradient = namedtuple('EmbeddingGradient', ['indices', 'values', 'dense_shape'])

  for layer in range(len(tvars)):
    if layer==0:
      new_dense_shape = gradient_sets[0][layer].dense_shape
      """
      NOTE: We do not use embedding layer gradients
      (see note below where we add IMM gradients to primary loss gradients)
      the following part of the code is hence being commented out to avoid performing
      accumulation of embedding layer gradients.
      Instead, we just return the first gradient (gradient_sets[0]) as a placeholder.

      accumulator = np.zeros(new_dense_shape)

      for idx, val in zip(
        [gradient_sets[grad_set][layer].indices for grad_set in range(num_grad_sets)],
        [gradient_sets[grad_set][layer].values  for grad_set in range(num_grad_sets)]
      ):
        accumulator[idx] += val

      idx = accumulator.sum(1).nonzero()[0]

      combined_grads[layer] = EmbeddingGradient(
        idx,
        accumulator[idx],
        new_dense_shape
      )
      """
      combined_grads[layer] = EmbeddingGradient(
        gradient_sets[0][layer].indices,
        gradient_sets[0][layer].values,
        new_dense_shape
      )
    else:
      combined_grads[layer] = np.sum(np.stack([
        gradient_sets[grad_set][layer] for grad_set in range(num_grad_sets)
      ]),0)

  return combined_grads


def run_epoch(epoch_ind, session, model, loader, split, update_op, flags,
        writer=None, summary_op=None, verbose=True, doIMM=False):
  """Run an epoch of training/testing"""
  epoch_size = loader.get_num_batches(split)
  start_time = time.time()
  total_cost = 0.0
  state = session.run(model._initializer)
  # state = session.run(model._initial_state)
  iters = 0

  vocab_size = len(loader.token_to_id)

  for k in range(1, epoch_size):
    doSampling = False

    if split=="train" and model.infer_only==False:
      # compute whether to do sampling this batch only if this is a training epoch
      if flags.sample_every > 0:
        doSampling = ((k % flags.sample_every) == (epoch_ind % flags.sample_every))

    if doIMM and doSampling:
      x, y, selected_long_histories, kn_y = \
        loader.get_batch(split, k, flags.unrolls_to_use, flags.long_hist_samples, True)
    else:
      x, y, kn_y = \
        loader.get_batch(split, k, flags.unrolls_to_use, flags.long_hist_samples, False)

    seq_len = [y.shape[1]] * flags.batch_size

    if split=="train" and model.infer_only==False:
      grad_imm = None

      """
      We compute the IMM gradients first and main (cross entropy) loss gradients later.
      This is because we will also update the state and print some information upon
      computing main loss.
      """
      if doIMM and doSampling:
        """
        This is the k-sample IMM, which is done every `FLAGS.sample_every` iteration
        """
        kn_y = np.reshape(kn_y, (-1, flags.unroll, vocab_size))
        kn_y = kn_y[:,-flags.unrolls_to_use:,:]

        grads_imm_per_unroll = []
        for i in range(flags.unrolls_to_use):
          kn_backoff_targets = kn_y[:, i, :]

          loss2_per_unroll = 0.0

          h = session.partial_run_setup(
            model.grads_imm_per_sample + model.q_y_out,
            model._x_imm_sampling + [
              model._initial_state_0_c, model._initial_state_0_h,
              model._initial_state_1_c, model._initial_state_1_h,
              model._y_imm_sampling, model._lookingAt
            ] + model.crosstalk_in
          )

          session.partial_run(h, [], feed_dict={
            # model._len: seq_len,
            model._initial_state_0_c: state[0].c,
            model._initial_state_0_h: state[0].h,
            model._initial_state_1_c: state[1].c,
            model._initial_state_1_h: state[1].h,
            model._lookingAt: i, # will be used to index the right timestep
            model._y_imm_sampling: kn_backoff_targets # this is already indexed
          })

          q_y = []
          for j in range(0, flags.long_hist_samples, flags.simultaneous_samples):

            feed_dict = {}
            for k in range(j, min(j+flags.simultaneous_samples, flags.long_hist_samples)):
              # `_x_imm_sampling[i]` takes (batch_size, unroll) input, i.e. `random_long_histories`
              # `_y_imm_sampling` takes (batch_size, vocab_size) output, i.e. `kn_backoff_targets`
              random_long_histories = \
                selected_long_histories[i*flags.batch_size : (i+1)*flags.batch_size,k,:]

              """
              NOTE: (pseudo) truncation is performed here
              `random_long_histories` is (20,35)
              We need to rotate `random_long_histories` to
              Bring the last `i` unroll positions to the first `i` (pseudo truncation)
              The remainder don't matter, and if we use `tf.nn.dynamic_rnn`, we will not
              be computing them.

              `flags.unroll - 1` is the max shift (for i=0)
              `flags.unroll - 1 - i` is the reduced shift (for i=flags.unrolls_to_use-1)
              """

              left_shift_by = flags.unroll - 1 - i
              pseudo_truncated_long = np.roll(random_long_histories, -left_shift_by, 1)

              feed_dict[model._x_imm_sampling[k]] = pseudo_truncated_long

            # NOTE: the `for j` and `for k` are essentially the same loop, "unflattened"
            # (we can perform computation for `FLAGS.simultaneous_samples` random samples together
            # this parameter depends on the computation device and should be set at the largest
            # possible value)
            q_y += session.partial_run(
              h,
              model.q_y_out[j:j+flags.simultaneous_samples],
              feed_dict=feed_dict
            )


          # Normalize the `q_y` using their sum. This is where the different
          # random samples "cross talk", through their sum
          sum_q_y = np.sum(np.stack(q_y),0)
          for j in range(flags.long_hist_samples):
            q_y[j] = q_y[j] / sum_q_y


          # Feed back the "cross-talking" `q_y` tensors and backprop
          # NOTE: again, the `for j` and `for k` are essentially the same loop, "unflattened"
          grads_imm_per_sample = []
          for j in range(0, flags.long_hist_samples, flags.simultaneous_samples):
            feed_dict = {}
            for k in range(j, min(j+flags.simultaneous_samples, flags.long_hist_samples)):
              # the `q_y` were normalized in place, so they are crosstalk tensors now
              feed_dict[model.crosstalk_in[k]] = q_y[k]

            grads_imm_per_sample += session.partial_run(
              h,
              model.grads_imm_per_sample[j:j+flags.simultaneous_samples],
              feed_dict=feed_dict
            )

          grads_imm_per_unroll.append(combine_gradients(grads_imm_per_sample, model.tvars))
          # pdb.set_trace()

        # The computed loss is not actual cross entropy. If printing IMM risk is
        # required, we can output an additional non-doctored cross entropy. See
        # the comments in the loss function.
        loss_imm = None
        grad_imm = combine_gradients(grads_imm_per_unroll, model.tvars)

      elif doIMM:
        """
        This is the single sample IMM variant, executed otherwise
        """
        feed_dict = {
          # model._len: seq_len,
          model._x_imm: x,
          model._y_imm: kn_y,
          model._initial_state_0_c: state[0].c,
          model._initial_state_0_h: state[0].h,
          model._initial_state_1_c: state[1].c,
          model._initial_state_1_h: state[1].h
        }

        loss_imm, grad_imm = session.run([model.loss_imm, model.grad_imm], feed_dict)


      """
      Compute main loss gradients and update the state.
      """
      fetches = [model.loss, model.grad if doIMM else model._train_op, model._final_state]

      feed_dict = {
        # model._len: seq_len,
        model._x: x,
        model._y: y,
        model._initial_state_0_c: state[0].c,
        model._initial_state_0_h: state[0].h,
        model._initial_state_1_c: state[1].c,
        model._initial_state_1_h: state[1].h
      }

      if summary_op is not None and writer is not None:
        fetches = [summary_op] + fetches
        summary, cost, grad, state = session.run(fetches, feed_dict)
        if k % flags.steps_per_summary == 0:
          writer.add_summary(summary, epoch_size*epoch_ind + k)
      else:
        cost, grad, state = session.run(fetches, feed_dict)

      """
      Accumulate IMM and main loss gradients.
      """
      if doIMM:
        feed_dict = {}

        # IMM component introduces quite a bit of annealing, specially when doing k-sample rather
        # than single sample IMM. Besides the embedding layer, the IMM component of the gradient
        # seems to not require gradient clipping.
        # Since we accumulate IMM gradients on CPU, incorporating gradient clipping will add an
        # overhead as we use a GPU version of TensorFlow which would require another data exchange
        # with the GPU.
        # Instead, we choose to simply not use embedding layer gradients when doing k-sample IMM
        # (we only use IMM gradients when doing single sample IMM).
        # The 2 factor on the primary gradients was experimentally determined to give more stable
        # gradients.
        feed_dict[model.gradientPlaceholders[model.tvars[0].name+"_indices"]] = \
          grad[0].indices if doSampling else np.hstack([grad[0].indices, grad_imm[0].indices])
        feed_dict[model.gradientPlaceholders[model.tvars[0].name+"_values"]] = \
          2 * grad[0].values if doSampling else np.vstack([2 * grad[0].values, grad_imm[0].values])
        for idx, tvar in enumerate(model.tvars[1:]):
          feed_dict[model.gradientPlaceholders[tvar.name]] = grad[idx+1] + grad_imm[idx+1]

        session.run(model._train_op_imm, feed_dict)


    else:
      fetches = [model.loss, model._final_state]

      feed_dict = {
        # model._len: seq_len,
        model._x: x,
        model._y: y,
        model._initial_state_0_c: state[0].c,
        model._initial_state_0_h: state[0].h,
        model._initial_state_1_c: state[1].c,
        model._initial_state_1_h: state[1].h
      }

      if summary_op is not None and writer is not None:
        fetches = [summary_op] + fetches
        summary, cost, state = session.run(fetches, feed_dict)
        if k % flags.steps_per_summary == 0:
          writer.add_summary(summary, epoch_size*epoch_ind + k)
      else:
        cost, state = session.run(fetches, feed_dict)

    total_cost += cost
    iters += flags.unroll

    if k % (epoch_size // flags.prints_per_epoch) == 10 and verbose:
      """
      Print the summary, including value of main cross entropy loss (exponentiated to perplexity).
      """
      log_info("%.3f perplexity: %.3f, speed: %.0f tps" %
        (
          k * 1.0 / epoch_size,
          np.exp(total_cost / iters),
          iters * flags.batch_size / (time.time() - start_time)
        )
      )

  return np.exp(total_cost / iters)

def main(_):
  if not os.path.exists(FLAGS.run_dir):
    os.makedirs(FLAGS.run_dir)
  file_handler = logging.FileHandler("{0}/log.txt".format(FLAGS.run_dir))
  logging.getLogger().addHandler(file_handler)

  DATA_PATHS = PTB_DATA_PATHS if FLAGS.dataset == "ptb" else TEXT8_DATA_PATHS
  log_info(str(DATA_PATHS))
  data_loader = TextLoader(DATA_PATHS, FLAGS.batch_size, FLAGS.unroll,
          FLAGS.token_type)
  vocab_size = len(data_loader.token_to_id)
  log_info("Vocabulary size: %d" % vocab_size)
  log_info(FLAGS.flag_values_dict())

  eval_flags = copy.deepcopy(FLAGS)
  eval_flags.batch_size = 1
  eval_flags.unroll = 1
  eval_flags.unrolls_to_use = 1

  tf.logging.set_verbosity(tf.logging.ERROR)
  with tf.Graph().as_default(), tf.Session() as session:
    initializer = \
      tf.random_uniform_initializer(-FLAGS.init_scale, FLAGS.init_scale, seed=FLAGS.seed)

    # Create training, validation, and evaluation models
    with tf.variable_scope("model", reuse=None, initializer=initializer):
      mtrain = LanguageModel(FLAGS, vocab_size, is_training=True)
    with tf.variable_scope("model", reuse=True, initializer=initializer):
      mvalid = LanguageModel(FLAGS, vocab_size, is_training=False)
      mtest = LanguageModel(eval_flags, vocab_size, is_training=False)

    summary_op = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.run_dir)
    model_saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=FLAGS.max_epochs)
    tf.global_variables_initializer().run()

    if FLAGS.restore_checkpoint is not None:
      saver = tf.train.import_meta_graph("{}.meta".format(FLAGS.restore_checkpoint))
      saver.restore(session, FLAGS.restore_checkpoint)

    lr = FLAGS.learning_rate
    decay_count = 0
    prev_valid_perplexity = None
    valid_perplexities = list()

    # for fine tuning, user should specify that IMM should be done from start
    # otherwise we will have to wait for validation perplexity to be computed (in first epoch)
    # before IMM can start from second epoch
    imm_started = FLAGS.imm_started
    if imm_started:
      FLAGS.prints_per_epoch = 100
    for k in range(FLAGS.max_epochs):
      mtrain.set_lr(session, lr)
      log_info("Epoch %d, learning rate %f" % (k, lr))

      train_perplexity = run_epoch(k, session, mtrain, data_loader, "train",
          mtrain._train_op, FLAGS, writer=train_writer, summary_op=summary_op, doIMM=imm_started)
      log_info("Epoch: %d Train Perplexity: %.3f" % (k, train_perplexity))
      valid_perplexity = run_epoch(k, session, mvalid, data_loader, "valid",
          tf.no_op(), FLAGS, verbose=False)
      log_info("Epoch: %d Valid Perplexity: %.3f" % (k, valid_perplexity))
      if valid_perplexity < FLAGS.start_imm_at:
        if not imm_started: # to make sure we don't end up bumping LR every time
          imm_started = True
          lr = 0.5 # bump up learning rate when IMM starts
          FLAGS.prints_per_epoch = 100
      if prev_valid_perplexity != None and\
              np.log(best_valid_perplexity) - np.log(valid_perplexity) < FLAGS.decay_threshold:
        lr = lr * FLAGS.learning_rate_decay
        decay_count += 1
        log_info("Loading epoch %d parameters, perplexity %f" %\
                (best_epoch, best_valid_perplexity))
        model_saver.restore(session, pjoin(FLAGS.run_dir, "model_epoch%d.ckpt" % best_epoch))
      prev_valid_perplexity = valid_perplexity

      valid_perplexities.append(valid_perplexity)
      if valid_perplexity <= np.min(valid_perplexities):
        best_epoch = k
        best_valid_perplexity = valid_perplexities[best_epoch]
        save_path = model_saver.save(
          session,
          pjoin(FLAGS.run_dir, "model_epoch%d.ckpt" % k),
          write_meta_graph=False
        )
        log_info("Saved model to file: %s" % save_path)

      if decay_count > FLAGS.max_decays:
        log_info("Reached maximum number of decays, quiting after epoch %d" % k)
        break

    if FLAGS.max_epochs > 0:
      log_info("Best perplexity %f achieved after epoch %d" % (best_valid_perplexity, best_epoch))
      if FLAGS.final:
        log_info("Loading as checkpoint for checking Test perplexity")
    
    if FLAGS.final:
      if FLAGS.max_epochs == 0:
        train_perplexity = \
          run_epoch(0, session, mvalid, data_loader, "train", tf.no_op(), FLAGS, verbose=True)
        log_info("Checkpoint Train Perplexity: %.3f" % train_perplexity)

        valid_perplexity = \
          run_epoch(0, session, mvalid, data_loader, "valid", tf.no_op(), FLAGS, verbose=False)
        log_info("Checkpoint Valid Perplexity: %.3f" % valid_perplexity)

      test_perplexity = \
        run_epoch(0, session, mvalid, data_loader, "test", tf.no_op(), FLAGS, verbose=False)
      log_info("Checkpoint Test Perplexity: %.3f" % test_perplexity)

      data_loader = \
        TextLoader(DATA_PATHS, eval_flags.batch_size, eval_flags.unroll, FLAGS.token_type)
      test_perplexity = \
        run_epoch(0, session, mtest, data_loader, "test", tf.no_op(), eval_flags, verbose=False)
      log_info("Repeat Test Perplexity with (batch_size, unroll) = (1,1): %.3f" % test_perplexity)

if __name__ == "__main__":
  tf.app.run()
