import torch
import torch.nn as nn


class ICM(nn.Module):
    def __init__(self, obs_dim, action_dim, feature_dim=288, hidden_dim=256,
        lr=1e-3):
        super().__init__()

        # s => phi(s)
        self.enc = nn.Linear(obs_dim, feature_dim)

        # (phi(s), phi(s')) => a_pred
        self.inv_model = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        # (phi(s), a) => s'_pred
        self.for_model = nn.Sequential(
            nn.Linear(feature_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )

        self.inv_criterion = nn.MSELoss()
        self.for_criterion = nn.MSELoss(reduction='none')
        self.inv_opt = torch.optim.Adam(
            list(self.enc.parameters()) + list(self.inv_model.parameters()),
            lr=lr)
        self.for_opt = torch.optim.Adam(self.for_model.parameters(), lr=lr)

    def inverse_dynamics(self, obs, next_obs, actions):
        '''
        obs, next_obs (M*N, D)
        actions (M*N, action_dim)
        '''
        # update inverse dynamics model
        obs_feat = self.enc(obs)
        next_obs_feat = self.enc(next_obs)
        a_pred = self.inv_model(torch.cat([obs_feat, next_obs_feat], dim=-1))

        loss = self.inv_criterion(a_pred, actions)

        self.inv_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.inv_opt.step()

        # return extracted features
        return obs_feat.detach(), next_obs_feat.detach()

    def forward_dynamics(self, obs_feat, next_obs_feat, actions):
        '''
        obs_feat, next_obs_feat (M*N, feature_dim)
        actions (M*N, action_dim)
        '''
        # predict next_obs using forward dynamics model
        next_obs_feat_pred = self.for_model(
            torch.cat([obs_feat, actions], dim=-1))

        # update model
        loss = self.for_criterion(next_obs_feat_pred, next_obs_feat)

        self.for_opt.zero_grad(set_to_none=True)
        (loss.mean()).backward()
        self.for_opt.step()

        # (M*N, feature_dim)
        return loss

    def forward(self, obs, next_obs, actions):
        # update inverse dynamics model and get features
        obs_feat, next_obs_feat = self.inverse_dynamics(obs, next_obs, actions)
        # predict next_obs using forward dynamics model and update model
        icm_loss = self.forward_dynamics(obs_feat, next_obs_feat, actions)
        icm_loss = icm_loss.detach().mean(dim=-1)
        return icm_loss
