## MIMEx: Intrinsic Rewards from Masked Input Modeling

This is the code for paper [MIMEx: Intrinsic Rewards from Masked Input Modeling](https://arxiv.org/abs/2305.08932). It contains the training code to reproduce all results on DeepMind Control Suite.

The RL code is built on top of [DrQ-v2](https://arxiv.org/abs/2107.09645).

### RL training code

Install [MuJoCo](http://www.mujoco.org/) if it is not already the case:

* Obtain a license on the [MuJoCo website](https://www.roboti.us/license.html).
* Download MuJoCo binaries [here](https://www.roboti.us/index.html).
* Unzip the downloaded archive into `~/.mujoco/mujoco200` and place your license key file `mjkey.txt` at `~/.mujoco`.
* Use the env variables `MUJOCO_PY_MJKEY_PATH` and `MUJOCO_PY_MUJOCO_PATH` to specify the MuJoCo license key path and the MuJoCo directory path.
* Append the MuJoCo subdirectory bin path into the env variable `LD_LIBRARY_PATH`.

Install the following libraries:
```sh
sudo apt update
sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3
```

Install dependencies:
```
conda env create -f conda_env.yml
conda activate mimex-dmc
```

### Example training commands

Train DrQ-v2 agent without additional exploration:
```
python train.py task=cartpole_swingup_sparse
```

Train DrQ-v2 agent with an exploration length of 5:
```
python train.py task=acrobot_swingup expl_cfg=expl expl_cfg.mask_ratio=0.8
```

Monitor results:
```
tensorboard --logdir exp_local
```

### Acknowledgments

We thank authors of DrQ-v2 for open-sourcing their code, and include their BibTex entry here:

```
@article{yarats2021drqv2,
  title={Mastering Visual Continuous Control: Improved Data-Augmented Reinforcement Learning},
  author={Denis Yarats and Rob Fergus and Alessandro Lazaric and Lerrel Pinto},
  journal={arXiv preprint arXiv:2107.09645},
  year={2021}
}
```
