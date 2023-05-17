#!/usr/bin/env python3

"""Actor critic."""

import re
import numpy as np
import os
import torch
import torch.nn as nn

from torch.distributions import MultivariateNormal

from mvp.backbones import vit


def schedule(schdl, step):
    try:
        return float(schdl)
    except ValueError:
        match = re.match(r'linear\((.+),(.+),(.+)\)', schdl)
        if match:
            init, final, duration = [float(g) for g in match.groups()]
            mix = np.clip(step / duration, 0.0, 1.0)
            return (1.0 - mix) * init + mix * final
        match = re.match(r'step_linear\((.+),(.+),(.+),(.+),(.+)\)', schdl)
        if match:
            init, final1, duration1, final2, duration2 = [
                float(g) for g in match.groups()
            ]
            if step <= duration1:
                mix = np.clip(step / duration1, 0.0, 1.0)
                return (1.0 - mix) * init + mix * final1
            else:
                mix = np.clip((step - duration1) / duration2, 0.0, 1.0)
                return (1.0 - mix) * final1 + mix * final2
    raise NotImplementedError(schdl)


###############################################################################
# States
###############################################################################

class ActorCritic(nn.Module):

    def __init__(
        self,
        obs_shape,
        states_shape,
        actions_shape,
        initial_std,
        encoder_cfg,
        policy_cfg,
        action_noise_cfg
    ):
        super(ActorCritic, self).__init__()
        assert encoder_cfg is None

        actor_hidden_dim = policy_cfg['pi_hid_sizes']
        critic_hidden_dim = policy_cfg['vf_hid_sizes']
        activation = nn.SELU()

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(*obs_shape, actor_hidden_dim[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dim)):
            if l == len(actor_hidden_dim) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dim[l], *actions_shape))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dim[l], actor_hidden_dim[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(*obs_shape, critic_hidden_dim[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dim)):
            if l == len(critic_hidden_dim) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dim[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dim[l], critic_hidden_dim[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        # print(self.actor)
        # print(self.critic)

        # Action noise
        if action_noise_cfg['type'] == "learned":
            self.log_std = nn.Parameter(
                np.log(initial_std) * torch.ones(*actions_shape))
        elif action_noise_cfg['type'] == "scheduled":
            self.action_noise_schedule = action_noise_cfg['schedule']
        elif action_noise_cfg['type'] == "none":
            self.log_std = nn.Parameter(
                torch.zeros(*actions_shape), requires_grad=False)
        else:
            raise ValueError(
                'Invalid action noise type', action_noise_cfg['type'])
        self.action_noise_type = action_noise_cfg['type']
        self.action_shape = actions_shape

        # Initialize the weights like in stable baselines
        actor_weights = [np.sqrt(2)] * len(actor_hidden_dim)
        actor_weights.append(0.01)
        critic_weights = [np.sqrt(2)] * len(critic_hidden_dim)
        critic_weights.append(1.0)
        self.init_weights(self.actor, actor_weights)
        self.init_weights(self.critic, critic_weights)

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    @torch.no_grad()
    def act(self, observations, states, step, eval_mode=False):
        actions_mean = self.actor(observations)

        if self.action_noise_type == "scheduled":
            # Use scheduled action noise
            stddev = schedule(self.action_noise_schedule, step)
            self.log_std = np.log(stddev) * torch.ones(*self.action_shape)
            self.log_std = self.log_std.to(states.device)

        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

        if eval_mode:
            actions = distribution.mean
        else:
            actions = distribution.sample()
        actions_log_prob = distribution.log_prob(actions)

        value = self.critic(observations)

        return (
            actions.detach(),
            actions_log_prob.detach(),
            value.detach(),
            actions_mean.detach(),
            self.log_std.repeat(actions_mean.shape[0], 1).detach(),
            None,  # dummy placeholder
        )

    @torch.no_grad()
    def act_inference(self, observations, states=None):
        actions_mean = self.actor(observations)
        return actions_mean

    def forward(self, observations, states, actions, step):
        actions_mean = self.actor(observations)

        if self.action_noise_type == "scheduled":
            # Use scheduled action noise
            stddev = schedule(self.action_noise_schedule, step)
            self.log_std = np.log(stddev) * torch.ones(*self.action_shape)
            self.log_std = self.log_std.to(states.device)

        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

        actions_log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()

        value = self.critic(observations)

        return actions_log_prob, entropy, value, actions_mean, self.log_std.repeat(actions_mean.shape[0], 1)


###############################################################################
# Pixels
###############################################################################

_HOI_MODELS = {
    "maevit-s16": "mae_pretrain_hoi_vit_small.pth",
}

_IN_MODELS = {
    "vit-s16": "sup_pretrain_imagenet_vit_small.pth",
    "maevit-s16": "mae_pretrain_imagenet_vit_small.pth",
}


class Encoder(nn.Module):

    def __init__(self, model_type, pretrain_dir, pretrain_type, freeze, emb_dim):
        super(Encoder, self).__init__()
        assert pretrain_type in ["imagenet", "hoi", "none"]
        if pretrain_type == "imagenet":
            assert model_type in _IN_MODELS
            pretrain_fname = _IN_MODELS[model_type]
            pretrain_path = os.path.join(pretrain_dir, pretrain_fname)
        elif pretrain_type == "hoi":
            assert model_type in _HOI_MODELS
            pretrain_fname = _HOI_MODELS[model_type]
            pretrain_path = os.path.join(pretrain_dir, pretrain_fname)
        else:
            pretrain_path = "none"
        assert pretrain_type == "none" or os.path.exists(pretrain_path)
        self.backbone, gap_dim = vit.vit_s16(pretrain_path)
        if freeze:
            self.backbone.freeze()
        self.freeze = freeze
        self.projector = nn.Linear(gap_dim, emb_dim)

    @torch.no_grad()
    def forward(self, x):
        feat = self.backbone.extract_feat(x)
        return self.projector(self.backbone.forward_norm(feat)), feat

    def forward_feat(self, feat):
        return self.projector(self.backbone.forward_norm(feat))


class PixelActorCritic(nn.Module):

    def __init__(
        self,
        obs_shape,
        states_shape,
        actions_shape,
        initial_std,
        encoder_cfg,
        policy_cfg,
        action_noise_cfg
    ):
        super(PixelActorCritic, self).__init__()
        assert encoder_cfg is not None

        # Encoder params
        model_type = encoder_cfg["model_type"]
        pretrain_dir = encoder_cfg["pretrain_dir"]
        pretrain_type = encoder_cfg["pretrain_type"]
        freeze = encoder_cfg["freeze"]
        emb_dim = encoder_cfg["emb_dim"]

        # Policy params
        actor_hidden_dim = policy_cfg["pi_hid_sizes"]
        critic_hidden_dim = policy_cfg["vf_hid_sizes"]
        activation = nn.SELU()

        # Obs and state encoders
        self.obs_enc = Encoder(
            model_type=model_type,
            pretrain_dir=pretrain_dir,
            pretrain_type=pretrain_type,
            freeze=freeze,
            emb_dim=emb_dim
        )
        self.state_enc = nn.Linear(states_shape[0], emb_dim)

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(emb_dim * 2, actor_hidden_dim[0]))
        actor_layers.append(activation)
        for li in range(len(actor_hidden_dim)):
            if li == len(actor_hidden_dim) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dim[li], *actions_shape))
            else:
                actor_layers.append(
                    nn.Linear(actor_hidden_dim[li], actor_hidden_dim[li + 1])
                )
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(emb_dim * 2, critic_hidden_dim[0]))
        critic_layers.append(activation)
        for li in range(len(critic_hidden_dim)):
            if li == len(critic_hidden_dim) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dim[li], 1))
            else:
                critic_layers.append(
                    nn.Linear(critic_hidden_dim[li], critic_hidden_dim[li + 1])
                )
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        # print(self.obs_enc)
        # print(self.state_enc)
        # print(self.actor)
        # print(self.critic)

        # Action noise
        if action_noise_cfg['type'] == "learned":
            self.log_std = nn.Parameter(
                np.log(initial_std) * torch.ones(*actions_shape))
        elif action_noise_cfg['type'] == "scheduled":
            self.action_noise_schedule = action_noise_cfg['schedule']
        elif action_noise_cfg['type'] == "none":
            self.log_std = nn.Parameter(
                torch.zeros(*actions_shape), requires_grad=False)
        else:
            raise ValueError(
                'Invalid action noise type', action_noise_cfg['type'])
        self.action_noise_type = action_noise_cfg['type']
        self.action_shape = actions_shape

        # Initialize the weights like in stable baselines
        actor_weights = [np.sqrt(2)] * len(actor_hidden_dim)
        actor_weights.append(0.01)
        critic_weights = [np.sqrt(2)] * len(critic_hidden_dim)
        critic_weights.append(1.0)
        self.init_weights(self.actor, actor_weights)
        self.init_weights(self.critic, critic_weights)

    @staticmethod
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(
                mod for mod in sequential if isinstance(mod, nn.Linear)
            )
        ]

    @torch.no_grad()
    def act(self, observations, states, step, return_seq_obs=None, eval_mode=False):
        obs_emb, obs_feat = self.obs_enc(observations)  # (B, emb_dim), (B, 384)
        state_emb = self.state_enc(states)  # (B, emb_dim)
        joint_emb = torch.cat([obs_emb, state_emb], dim=1)

        actions_mean = self.actor(joint_emb)

        if self.action_noise_type == "scheduled":
            # Use scheduled action noise
            stddev = schedule(self.action_noise_schedule, step)
            self.log_std = np.log(stddev) * torch.ones(*self.action_shape)
            self.log_std = self.log_std.to(states.device)

        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

        if eval_mode:
            actions = distribution.mean
        else:
            actions = distribution.sample()
        actions_log_prob = distribution.log_prob(actions)

        value = self.critic(joint_emb)

        if return_seq_obs is None:
            seq_obs = None
        elif return_seq_obs == "obs_feat":
            seq_obs = obs_feat.detach()
        elif return_seq_obs == "obs_emb":
            seq_obs = obs_emb.detach()
        elif return_seq_obs == "joint_emb":
            seq_obs = joint_emb.detach()
        else:
            raise ValueError("Invalid return seq obs type:", return_seq_obs)

        return (
            actions.detach(),
            actions_log_prob.detach(),
            value.detach(),
            actions_mean.detach(),
            self.log_std.repeat(actions_mean.shape[0], 1).detach(),
            obs_feat.detach(),  # return obs features
            seq_obs
        )

    @torch.no_grad()
    def act_inference(self, observations, states):
        obs_emb, _ = self.obs_enc(observations)
        state_emb = self.state_enc(states)
        joint_emb = torch.cat([obs_emb, state_emb], dim=1)
        actions_mean = self.actor(joint_emb)
        return actions_mean

    def forward(self, obs_features, states, actions, step):
        obs_emb = self.obs_enc.forward_feat(obs_features)
        state_emb = self.state_enc(states)
        joint_emb = torch.cat([obs_emb, state_emb], dim=1)

        actions_mean = self.actor(joint_emb)

        if self.action_noise_type == "scheduled":
            # Use scheduled action noise
            stddev = schedule(self.action_noise_schedule, step)
            self.log_std = np.log(stddev) * torch.ones(*self.action_shape)
            self.log_std = self.log_std.to(actions.device)

        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

        actions_log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()

        value = self.critic(joint_emb)

        return (
            actions_log_prob,
            entropy,
            value,
            actions_mean,
            self.log_std.repeat(actions_mean.shape[0], 1),
        )
