import numpy as np
import torch
import torch.nn as nn
from torch.nn import init


class RND(nn.Module):
    def __init__(self, obs_dim, feature_dim=288, hidden_dim=256, lr=1e-3):
        super().__init__()
        self.target = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )
        self.predictor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )

        self.criterion = nn.MSELoss(reduction='none')
        self.opt = torch.optim.Adam(self.predictor.parameters(), lr=lr)

        # parameter initialization following original paper
        for p in self.modules():
            if isinstance(p, nn.Linear):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

        # freeze target network
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, obs):
        target_feat = self.target(obs)
        pred_feat = self.predictor(obs)

        # update model
        loss = self.criterion(pred_feat, target_feat)
        self.opt.zero_grad(set_to_none=True)
        (loss.mean()).backward()
        self.opt.step()

        # (M*N, feature_dim) => (M*N,)
        return loss.detach().mean(dim=-1)
