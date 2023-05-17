#!/usr/bin/env python3

"""Train a policy with PPO."""

import hydra
import omegaconf
import os
import datetime

import wandb

from mvp.utils.hydra_utils import omegaconf_to_dict, print_dict, dump_cfg
from mvp.utils.hydra_utils import set_np_formatting, set_seed
from mvp.utils.hydra_utils import parse_sim_params, parse_task
from mvp.utils.hydra_utils import process_ppo


def preprocess_cfg(cfg):
    # just to make things easier...
    cfg.sim_device = f'cuda:{cfg.graphics_device_id}'
    cfg.rl_device = f'cuda:{cfg.graphics_device_id}'

    # automatic folder naming
    curr_time = str(datetime.datetime.now())[:19].replace(' ', '_')

    if cfg.expl_cfg.expl_seq_len == 1:
        assert cfg.expl_cfg.baseline == "rnd"

    id_args = [
        ['', curr_time],
        ['l', cfg.expl_cfg.expl_seq_len],
        ['k', cfg.expl_cfg.k_expl],
    ]

    # exploration configs
    if cfg.expl_cfg.expl_seq_len >= 2 and cfg.expl_cfg.baseline == "none":
        id_args += [
            ['mr', cfg.expl_cfg.mask_ratio],
        ]
    if cfg.expl_cfg.baseline != "none":
        id_args += [['', cfg.expl_cfg.baseline]]

    id_args += [['s', cfg.train.seed]]

    cfg_id = '_'.join([f'{n}{v}' for n, v in id_args])
    exp_name = '{}/{}_{}'.format(cfg.exp_name, cfg.task.name, cfg_id)
    cfg.logdir = os.path.join(cfg.logdir, exp_name)
    print('LOGDIR ===> ', cfg.logdir)

    if cfg.task.name[-6:] == 'Sparse':
        # remove the "Sparse" str
        cfg.task.name = cfg.task.name[:-6]

    return cfg


@hydra.main(config_name="config", config_path="../configs")
def train(cfg: omegaconf.DictConfig):

    # Assume no multi-gpu training
    assert cfg.num_gpus == 1

    cfg = preprocess_cfg(cfg)

    # Parse the config
    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    wandb.init(
        project="mimex", name=cfg.logdir.split("/")[-1], config=cfg_dict,
        mode=cfg.wandb_mode)

    # Create logdir and dump cfg
    if not cfg.test:
        os.makedirs(cfg.logdir, exist_ok=True)
        dump_cfg(cfg, cfg.logdir)

    # Set up python env
    set_np_formatting()
    set_seed(cfg.train.seed, cfg.train.torch_deterministic)

    # Construct task
    sim_params = parse_sim_params(cfg, cfg_dict)
    env = parse_task(cfg, cfg_dict, sim_params)

    # Perform training
    ppo = process_ppo(env, cfg, cfg_dict, cfg.logdir, cfg.cptdir)
    ppo.run(num_learning_iterations=cfg.train.learn.max_iterations, log_interval=cfg.train.learn.save_interval)

    wandb.finish()


if __name__ == '__main__':
    train()
