r"""
This `loader.py` file is adapted from the codebase of our main baseline
    Data Noising as Smoothing in Neural Network Language Models (ICLR 2017)
    https://github.com/stanfordmlgroup/nlm-noising

which in turn was adapted from
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/ptb/reader.py

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

import collections

# from utils import bigram_counts, trigram_counts, build_continuations
# from utils import estimate_modkn_discounts
from random import randrange

import numpy as np
from scipy.sparse import lil_matrix


def _read_tokens(filename, level="word"):
  with open(filename, "r") as f:
    if "ptb" in filename:
      tokens = f.read().replace("\n", "<eos>")
    elif "text8" in filename:
      tokens = f.read().strip()
    else:
      assert(False)
    if level == "word":
      tokens = tokens.split()
    return tokens


def _file_to_token_ids(filename, token_to_id, level):
  data = _read_tokens(filename, level=level)
  return data, [token_to_id[token] for token in data]


def _build_vocab(filename, level):
  data = _read_tokens(filename, level=level)
  counter = collections.Counter(data)
  # Use this to get tokens sorted by frequencies
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
  total_count = sum(counter.values())
  frequencies = dict((k, v / float(total_count))
                     for k, v in counter.items())

  # Compute number of different histories
  bg_hist_sets = collections.defaultdict(set)
  for k in range(1, len(data)):
    bg_hist_sets[data[k]].add(data[k - 1])
  bg_hist_counts = dict([(k, len(s)) for k, s in bg_hist_sets.items()])
  # NOTE Edge case here where first word never appears again
  if data[0] not in bg_hist_counts:
    bg_hist_counts[data[0]] = 1
  total_hists = sum(bg_hist_counts.values())

  tokens, _ = list(zip(*count_pairs))
  token_to_id = dict(zip(tokens, range(len(tokens))))
  sorted_frequencies = [frequencies[token] for token in tokens]
  sorted_hist_freqs = [bg_hist_counts[token] /
                       float(total_hists) for token in tokens]

  return token_to_id, sorted_frequencies, sorted_hist_freqs


def _reshape_data(raw_data, batch_size, unroll):
  data_len = len(raw_data)
  batch_len = data_len // batch_size
  data = np.zeros([batch_size, batch_len], dtype=np.int32)
  for i in range(batch_size):
    data[i] = raw_data[batch_len * i:batch_len * (i + 1)]
  epoch_size = batch_len // unroll
  if epoch_size == 0:
    raise ValueError("epoch_size == 0, decrease batch_size or unroll")
  return data


def load_text_data(data_paths, level):
  assert len(data_paths) == 3
  train_path = data_paths[0]
  valid_path = data_paths[1]
  test_path = data_paths[2]

  token_to_id, frequencies, hist_freqs = _build_vocab(train_path, level)
  train_tokens, train_data = _file_to_token_ids(train_path, token_to_id, level)
  _, valid_data = _file_to_token_ids(valid_path, token_to_id, level)
  _, test_data = _file_to_token_ids(test_path, token_to_id, level)

  return train_data, valid_data, test_data, token_to_id, frequencies, hist_freqs, train_tokens

class TextLoader(object):

  def __init__(self, data_paths, batch_size, unroll, level):
    self.batch_size = batch_size
    self.unroll = unroll
    train_data, valid_data, test_data, token_to_id, frequencies, hist_freqs, train_tokens = \
      load_text_data(data_paths, level)

    self.longFromShort = {
      "train": {},
      "valid": {},
      "test": {}
    }
    for (split, dataset) in zip(["train", "valid", "test"], [train_data, valid_data, test_data]):
      for i in range(unroll,len(dataset)+1):
        chunk = dataset[slice(i-unroll,i)]

        if chunk[-1] in self.longFromShort[split]:
          self.longFromShort[split][chunk[-1]].append(chunk)
        else:
          self.longFromShort[split][chunk[-1]] = [chunk]

      for k in self.longFromShort[split]:
        self.longFromShort[split][k] = np.array(self.longFromShort[split][k])


    self.token_to_id = token_to_id

    self.vocab_size = len(self.token_to_id)














    """
    Construct bigram counts matrix
    """
    print("Computing Kneser-Ney bigram of the corpus")
    self.C_bigram = lil_matrix((self.vocab_size, self.vocab_size), dtype=np.int32)

    # Initialize counters for bigrams that occur exactly 1, 2, and 3 times
    N1 = 0
    N2 = 0
    N3 = 0
    N4 = 0

    # Fill the lil_matrix and count bigrams with occurrences of 1, 2, 3 and 4
    for (i, j), count in collections.Counter(zip(train_data[:-1], train_data[1:])).items():
        self.C_bigram[i, j] = count

        if count == 1:
            N1 += 1
        elif count == 2:
            N2 += 1
        elif count == 3:
            N3 += 1
        elif count == 4:
            N4 += 1

    # Estimate discounting parameters
    Y = N1 / (N1 + 2 * N2)
    self.D1 = 1 - 2 * Y * (N2 / N1)
    self.D2 = 2 - 3 * Y * (N3 / N2)
    self.D3p = 3 - 4 * Y * (N4 / N3)

    # Get a floating point counts matrix
    # We will discount it before normalizing it to get the probabilities
    self.P_bigram = self.C_bigram.astype(np.float64).toarray()

    # Get missing mass per row using discount factors
    print("\tComputing missing mass")
    self.gamma_row = np.empty((1, self.vocab_size))
    for row_idx in range(self.vocab_size):
      # if row_idx % 500 == 0:
      #   print(row_idx)
      row = self.C_bigram.getrowview(row_idx)
      self.gamma_row[0,row_idx] = self.D1 * (row == 1).getnnz() + \
        self.D2 * (row == 2).getnnz() + self.D3p * (row >= 3).getnnz()
      self.gamma_row[0,row_idx] /= row.sum()
    print("\t...done")

    # Get row wise sums before discounting
    c_all = np.sum(self.P_bigram, 1)

    # Do the discounting
    self.P_bigram[(self.C_bigram == 1).nonzero()] -= self.D1
    self.P_bigram[(self.C_bigram == 2).nonzero()] -= self.D2
    self.P_bigram[(self.C_bigram >= 3).nonzero()] -= self.D3p

    # Normalize
    self.P_bigram /= np.expand_dims(c_all,1)

    # Generate the (unigram) KN backoff (probability of continuation)
    self.p_cont = (self.C_bigram > 0).sum(0) / (self.C_bigram > 0).sum()

    # Overlay probability of continuation on the discounted+normalized counts
    # After weighing them with row-wise gamma (missing mass)
    self.P_bigram += (self.gamma_row.T * self.p_cont)
    print("...done")


    # NOTE: this extends the vocabulary
    self.token_to_id['<_>'] = len(self.token_to_id)
    self.id_to_token = dict((v, k) for k, v in self.token_to_id.items())

    # need to update because we have extended the vocabulary
    self.P_bigram = np.hstack([self.P_bigram, np.zeros((self.vocab_size, 1))])
    self.vocab_size = len(self.token_to_id)


    # we will be picking (20*35) rows of `self.P_bigram` in every `get_batch` call
    # the last one corresponding to '<_>' will never be picked
    # because no x has it in the corresponding y
    # therefore, no need to do `np.vstack` on `self.P_bigram`
    # last column can remain zero, it will still sum to 1


    train_data = _reshape_data(train_data, batch_size, unroll)
    valid_data = _reshape_data(valid_data, batch_size, unroll)
    test_data = _reshape_data(test_data, batch_size, unroll)
    self.split_data = {"train": train_data, "valid": valid_data,
                       "test": test_data}
    self.frequencies = frequencies
    self.frequencies_cumsum = np.cumsum(frequencies)
    self.hist_freqs = hist_freqs
    self.hist_freqs_cumsum = np.cumsum(hist_freqs)

  def get_num_batches(self, split):
    return (self.split_data[split].shape[1] - 1) // self.unroll

  def get_batch(self, split, index, unrolls_to_use, long_hist_samples, doSampling=False):
    split_data = self.split_data[split]
    i = index
    x = split_data[:, i * self.unroll:(i + 1) * self.unroll]
    y = split_data[:, i * self.unroll + 1:(i + 1) * self.unroll + 1]

    # this will contain a random history sample for each unroll position
    selected_long_histories = np.zeros((0,self.unroll), dtype=int)

    num_samples = long_hist_samples


    # (batch_size, unrolls_to_use, kn_based_dist)
    kn_y = np.zeros((x.shape[0], self.unroll, self.vocab_size))


    for i in range(self.batch_size):
      for j in range(self.unroll):
        if doSampling:
          # use true KN bigram and backoff
          kn_y[i, j] = self.P_bigram[x[i, j]]
        else:
          # use one-hot targets
          target = y[i, j],
          kn_y[i, j, target] = 1

    # Get discount factors
    gamma = self.gamma_row[:,x.flatten()].T

    kn_y = np.reshape(kn_y, (-1, self.vocab_size))

    if not doSampling:
      # we still have to mix the KN Backoff into the one-hot targets
      kn_y = np.multiply(1 - gamma, kn_y) + np.matmul(gamma, np.hstack([self.p_cont, [[0]]]))
    
    # right now this is (batch_size * unrolls_to_use) vectors each of length `vocab_size`
    kn_y = np.asarray(kn_y)

    if not doSampling:
      return x, y, kn_y


    # the order, i.e. 'F' or 'C' depends on how we handle it in the loop in `run_epoch`
    
    # first, collect randomly sampled histories
    for short_history in x[:,-unrolls_to_use:].flatten(order='F'):
      
      long_histories = self.longFromShort[split][short_history]
      num_long_histories = len(long_histories)

      # generate `selected_indices`
      selected_indices = []
      for k in range(num_samples):
        random_index = randrange(num_long_histories)
        selected_indices.append(random_index)

      for random_index in selected_indices:
        rand_long_history = np.expand_dims(long_histories[random_index],0)
        selected_long_histories = np.vstack([selected_long_histories, rand_long_history])

    selected_long_histories = np.reshape(selected_long_histories, (-1, num_samples, self.unroll))
    # selected_long_histories = np.moveaxis(selected_long_histories, 0, 1)

    return x, y, selected_long_histories, kn_y




if __name__ == "__main__":
  from cfg import PTB_DATA_PATHS
  loader = TextLoader(PTB_DATA_PATHS, 20, 35, "word")

  print("most frequent token: %s" %
        loader.id_to_token[np.argmax(loader.frequencies)])
  print("token with most distinct histories: %s" %
        loader.id_to_token[np.argmax(loader.hist_freqs)])
  print("tokens with most distinct continuations: %s" % sorted(loader.continuations[
        "distinct"].iterkeys(), key=(lambda key: -loader.continuations["distinct"][key]))[0:10])
  print("tokens with most total continuations: %s" % sorted(loader.continuations[
        "total"].iterkeys(), key=(lambda key: -loader.continuations["total"][key]))[0:10])
