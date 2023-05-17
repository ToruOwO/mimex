#!/usr/bin/env python3

"""Rollout storage."""

import torch

from torch.utils.data.sampler import BatchSampler
from torch.utils.data.sampler import SequentialSampler
from torch.utils.data.sampler import SubsetRandomSampler

from mvp.ppo.bert import BERT, MAE
from mvp.ppo.icm import ICM
from mvp.ppo.rnd import RND


class RolloutStorage:

    def __init__(
        self,
        num_envs,
        num_transitions_per_env,
        obs_shape,
        states_shape,
        actions_shape,
        expl_cfg,
        device="cpu",
        sampler="sequential"
    ):

        self.device = device
        self.sampler = sampler

        # Core
        self.observations = None
        self.states = torch.zeros(num_transitions_per_env, num_envs, *states_shape, device=self.device)
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()

        # For PPO
        self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)

        self.explore = (expl_cfg["expl_seq_len"] >= 1)
        self.baseline = expl_cfg["baseline"]
        if self.explore:
            # Optional exploration bonus
            self.expl_r = torch.zeros_like(self.rewards, device=self.device)
            self.k_expl = expl_cfg["k_expl"]
            self.expl_seq_len = expl_cfg["expl_seq_len"]
            self.seq_obs = torch.zeros(
                num_transitions_per_env, num_envs, self.expl_seq_len,
                expl_cfg["input_dim"], device=self.device
            )

            if expl_cfg["baseline"] == "icm":
                # NOTE: assuming action has only 1 dim
                self.icm = ICM(
                    expl_cfg["input_dim"], actions_shape[-1]).to(self.device)
            elif expl_cfg["baseline"] == "rnd":
                self.rnd = RND(expl_cfg["input_dim"]).to(self.device)
            elif expl_cfg["baseline"] == "avg":
                self.avg_iter = 0
                self.obs_avg = torch.zeros(
                    self.expl_seq_len, expl_cfg["input_dim"],
                    device=self.device)
            else:
                self.n_mask = expl_cfg["n_mask"]

                if expl_cfg["mask_all"]:
                    self.bert = MAE(
                        seq_len=self.expl_seq_len,
                        feature_dim=expl_cfg["input_dim"],
                        embed_dim=expl_cfg["embed_dim"],
                        decoder_embed_dim=expl_cfg["decoder_embed_dim"],
                        decoder_num_heads=expl_cfg["decoder_num_heads"],
                        decoder_depth=expl_cfg["decoder_depth"],
                        mask_ratio=expl_cfg["mask_ratio"],
                        norm_loss=expl_cfg["norm_loss"],
                        use_cls=expl_cfg["use_cls"],
                        ).to(self.device)
                else:
                    self.bert = BERT(
                        seq_len=self.expl_seq_len,
                        feature_dim=expl_cfg["input_dim"],
                        embed_dim=expl_cfg["embed_dim"],
                        decoder_embed_dim=expl_cfg["decoder_embed_dim"],
                        decoder_num_heads=expl_cfg["decoder_num_heads"],
                        decoder_depth=expl_cfg["decoder_depth"],
                        mask_ratio=expl_cfg["mask_ratio"],
                        norm_loss=expl_cfg["norm_loss"],
                        use_cls=expl_cfg["use_cls"],
                        ).to(self.device)
                self.bert_opt = torch.optim.Adam(
                    self.bert.parameters(), lr=expl_cfg["bert_lr"])

        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        self.step = 0

    def add_transitions(
        self, observations, states, actions, rewards, dones, values,
        actions_log_prob, mu, sigma, seq_obs):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")

        if self.observations is None:
            self.observations = torch.zeros(
                self.num_transitions_per_env, self.num_envs,
                *observations.shape[1:], device=self.device
            )
        self.observations[self.step].copy_(observations)
        self.states[self.step].copy_(states)
        self.actions[self.step].copy_(actions)
        self.rewards[self.step].copy_(rewards.view(-1, 1))
        self.dones[self.step].copy_(dones.view(-1, 1))
        self.values[self.step].copy_(values)
        self.actions_log_prob[self.step].copy_(actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(mu)
        self.sigma[self.step].copy_(sigma)

        if self.explore:
            self.seq_obs[self.step].copy_(seq_obs)

        self.step += 1

    def clear(self):
        self.step = 0

    def compute_expl_r(self, anneal_weight=1.0):
        if not self.explore:
            return 0.

        M, N, T, D = self.seq_obs.shape

        if self.baseline == "rnd":
            # compute intrinsic reward using RND
            obs = self.seq_obs[:, :, 0].view(M * N, D)
            rnd_loss = self.rnd(obs)
            self.expl_r = rnd_loss.view(*self.rewards.shape)
        elif self.baseline == "icm":
            # compute intrinsic reward using ICM
            # obs, next_obs, actions
            obs = self.seq_obs[:, :, 0].view(M * N, D)
            next_obs = self.seq_obs[:, :, 1].view(M * N, D)
            actions = self.actions.view(M * N, -1)  # (M, N, action_dim)

            # update inverse dynamics model and get features
            # predict next_obs using forward dynamics model and update model
            icm_loss = self.icm(obs, next_obs, actions)  # (M*N,)

            # relabel rewards
            self.expl_r = icm_loss.view(*self.rewards.shape)
        elif self.baseline == "avg":
            # calculate exploration reward from moving average
            # (M, N, T, D) => (M, N, 1)
            self.expl_r = ((self.seq_obs - self.obs_avg) ** 2).mean(
                dim=-1).mean(dim=-1, keepdim=True)

            # update moving average of obs embeddings
            self.avg_iter += 1
            avg = self.seq_obs.view(M * N, T, D).mean(dim=0)
            self.obs_avg = self.obs_avg - (self.obs_avg - avg) / self.avg_iter
        else:

            bert_input = self.seq_obs.view(M * N, T, D)

            # calculate BERT loss
            # (M, N, T, *obs_shape)
            bert_loss, _, _ = self.bert(bert_input, keep_batch=True)
            if self.n_mask > 1:
                # mask multiple times and calculate average to reduce variance
                for _ in range(self.n_mask - 1):
                    l, _, _ = self.bert(bert_input, keep_batch=True)
                    bert_loss += l
                bert_loss /= self.n_mask

            # update BERT
            self.bert_opt.zero_grad(set_to_none=True)
            (bert_loss.mean()).backward()
            self.bert_opt.step()

            # relabel rewards
            self.expl_r = bert_loss.detach().view(M, N, 1)

        self.rewards += self.expl_r * self.k_expl * anneal_weight

        return self.expl_r.mean()

    def compute_returns(self, last_values, gamma, lam):
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            self.returns[step] = advantage + self.values[step]

        # Compute and normalize the advantages
        self.advantages = self.returns - self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def get_statistics(self):
        done = self.dones.cpu()
        done[-1] = 1
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        done_indices = torch.cat(
            (flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero(as_tuple=False)[:, 0])
        )
        trajectory_lengths = (done_indices[1:] - done_indices[:-1])
        return trajectory_lengths.float().mean(), self.rewards.mean()

    def mini_batch_generator(self, num_mini_batches):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches

        if self.sampler == "sequential":
            # For physics-based RL, each environment is already randomized. There is no value to doing random sampling
            # but a lot of CPU overhead during the PPO process. So, we can just switch to a sequential sampler instead
            subset = SequentialSampler(range(batch_size))
        elif self.sampler == "random":
            subset = SubsetRandomSampler(range(batch_size))

        batch = BatchSampler(subset, mini_batch_size, drop_last=True)
        return batch
