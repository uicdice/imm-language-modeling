## IMM in LSTM RNN

This is the second out of the three repositories accompanying the paper *Induced Model Matching: Restricted Models Help Train Full-Featured Models (NeurIPS 2024)*

```bibtex
@inproceedings{muneeb2024induced,
    title     = {Induced Model Matching: Restricted Models Help Train Full-Featured Models},
    author    = {Usama Muneeb and Mesrob I Ohannessian},
    booktitle = {The Thirty-eighth Annual Conference on Neural Information Processing Systems},
    year      = {2024},
    url       = {https://openreview.net/forum?id=iW0wXE0VyR}
}
```

This repository demonstrates Induced Model Matching (IMM) in the training of an LSTM RNN language model by matching its induced bigram against the Kneser-Ney bigram of the dataset.

Other repositories: [IMM in Logistic Regression](https://github.com/uicdice/imm-logistic-regression) | [IMM in learning MDPs (REINFORCE)](https://github.com/uicdice/imm-reinforce)

This code is based on Xie et al's [nlm-noising](https://github.com/stanfordmlgroup/nlm-noising) code. Like the original, it is being released under Apache 2.0 license.

**Compatibility**: This code will run on both TensorFlow 1 and TensorFlow 2 (in compatibility mode). It has currently **not** been tested on NumPy 2.0 and Keras 3.0 is not supported.

If you are using Nvidia containers, `nvcr.io/nvidia/tensorflow:24.03-tf2-py3` (dated March 2024) is confirmed to work. If you are using the official TensorFlow container (CPU only), `tensorflow/tensorflow:2.15.0` (dated December 2023) is confirmed to work.

### Training from scratch

For **PTB** dataset, we use a model size of 1500 and a dropout probability 0.65. We start doing IMM when validation perplexity reaches below 90.

```bash
python lm.py --run_dir $HOME/ptb_pretrain_1500  --hidden_dim 1500 --drop_prob 0.65 --dataset ptb
```

> [!NOTE]
> **Expected behaviour**: Ideally the training should converge to within a validation perplexity of 90 within the first 15 epochs of training (and consequently the IMM regularization should start). If this happens, the run will most likely be successful and go all the way to 76.x.

> [!CAUTION]
> **Known issues**: The code that performs the training before introduction of IMM is the same as the baseline code (i.e. [nlm-noising](https://github.com/stanfordmlgroup/nlm-noising)). Just like the baseline, when training on PTB, it may get stuck in a local optimum when the train and validation perplexities are just below a perplexity of 700. If this happens, please terminate and restart training. In some cases, NaN's may be observed which also indicates that the run has failed and needs a restart. To help you bypass both of these issues, we have provided a [pretrained checkpoint](https://drive.google.com/drive/folders/1ExVV9C0Ito_Dj3yEwl53vPxFljsTz0Pg?usp=sharing) which you can start with and then fine tune with IMM (see below).

### Option to resume training a checkpoint from the baseline code

If you have a model trained using Xie et al's [nlm-noising](https://github.com/stanfordmlgroup/nlm-noising) code, you may as well load it as a checkpoint and and further train using IMM as below. 

> [!NOTE]
> If you use this option, please use a checkpoint trained using pure dropout and no noising.
> 
> In addition, please use a partially converged checkpoint (i.e. one with a slightly higher perplexity than the best achievable using pure dropout as pointed out in Table 2 and 3 of [Xie et al](https://arxiv.org/pdf/1703.02573.pdf)). The best achievable for PTB is 81.6.

For **PTB** dataset

```bash
python lm.py --run_dir $HOME/ptb_finetune_1500 --hidden_dim 1500 --drop_prob 0.65 --dataset ptb --lambda_param 0.2 --long_hist_samples 10 --restore_checkpoint $HOME/ptb_pretrain_1500/model_epoch38.ckpt --learning_rate 0.5 --learning_rate_decay 0.5 --max_decays 20 --imm_started
```

(please replace `model_epoch38` with the checkpoint you would like to load)
