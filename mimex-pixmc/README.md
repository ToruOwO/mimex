## MIMEx: Intrinsic Rewards from Masked Input Modeling

This is the code for paper [MIMEx: Intrinsic Rewards from Masked Input Modeling](https://arxiv.org/abs/2305.08932). It contains the PixMC-Sparse benchmark suite and the training code to reproduce all results on PixMC-Sparse environment.

The RL code is built on top of [MVP](https://arxiv.org/abs/2203.06173).

### Pre-trained visual enocoders

We use the pre-trained visual encoder available from MVP [repo](https://github.com/ir413/mvp), in particular the ViT-S model pretrained with HOI dataset. Follow [this](https://www.dropbox.com/s/51fasmar8hjfpeh/mae_pretrain_hoi_vit_small.pth?dl=1) link to download the pretrained encoder.

### Benchmark suite and RL training code

Please see [`INSTALL.md`](INSTALL.md) for installation instructions.

### Example training commands

Train `KukaPickSparse` with an exploration length of 5:

```
python tools/train.py task=KukaPickSparsePixels expl_cfg=expl_l5 expl_cfg.k_expl=0.5
```

Train `KukaPickSparse` with an exploration length of 5 and mask for 5 times:

```
python tools/train.py task=KukaPickSparsePixels expl_cfg=expl_l5 expl_cfg.k_expl=0.5 expl_cfg.n_mask=5
```

Train `KukaPickSparse` with a larger Transformer decoder:

```
python tools/train.py task=KukaPickSparsePixels expl_cfg=expl_l5 expl_cfg.k_expl=0.5 expl_cfg.decoder_embed_dim=128 expl_cfg.decoder_num_heads=4 expl_cfg.decoder_depth=2
```

### Acknowledgments

We thank authors of MVP for open-sourcing their code and pre-trained models, and include their BibTex entry here:

```
@article{Xiao2022
  title = {Masked Visual Pre-training for Motor Control},
  author = {Tete Xiao and Ilija Radosavovic and Trevor Darrell and Jitendra Malik},
  journal = {arXiv:2203.06173},
  year = {2022}
}
```
