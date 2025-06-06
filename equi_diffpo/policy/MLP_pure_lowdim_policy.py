import einops
from einops import rearrange, reduce
from zero.z_utils.coding import extract
from zero.FrankaPandaFK_torch import FrankaEmikaPanda_torch
from codebase.z_utils.Rotation_torch import PosEuler2HT, HT2eePose
from codebase.z_utils.Rotation_torch import matrix_to_rotation_6d, eePose2HT
import numpy as np
from typing import Dict, Tuple
import torch
from equi_diffpo.model.common.rotation_transformer import RotationTransformer
from equi_diffpo.model.common.normalizer import LinearNormalizer
from equi_diffpo.policy.base_image_policy import BaseImagePolicy
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms

import IPython
e = IPython.embed
"""
Policy的要求有哪些

有def compute_loss(batch)->loss
有def predict_action(obs_dict)->action_dict
有self.normalizer: LinearNormalizer

接收horizon的
batch的
obs(B,n_obs,...)
action(B,horizon,...)

return loss of all horizon actions

"""


class SingleHeadMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # two hidden layers → final output dim = 1 (a single action value)
        self.net = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8)  # Assuming the output is a 6D vector (e.g., rotation)
        )

    def forward(self, x):
        # x: [batch_size, state_dim] → returns [batch_size, 1]
        return self.net(x)


class MLPModel(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        activation = nn.ReLU
        layer_dims = [90, 128, 256, 512, 512, 256, 256, 128]
        activate_last = False  # default is False, can be overridden by args_override

        for in_dim, out_dim in zip(layer_dims[:-1], layer_dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            # apply activation after every layer except, by default, the last
            layers.append(activation())
        if not activate_last:
            # remove the final activation
            layers.pop()
        self.encoder = nn.Sequential(*layers)

        self.heads = nn.ModuleList([
            SingleHeadMLP()
            for _ in range(16)
        ])

        # 16 mlp decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        outs = [head(x) for head in self.heads]
        stacked = torch.stack(outs, dim=1)  # [B, 16, 10]
        assert stacked.shape[1] == 16, f"Expected 16 heads, got {stacked.shape[1]}"
        return stacked


class MLPPolicy(BaseImagePolicy):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = MLPModel()
        self.normalizer = LinearNormalizer()

    def compute_loss(self, batch):  # 默认是两个obs进来吧

        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])

        states = nobs['states']
        states = einops.rearrange(states, 'b h d -> b (h d)')
        action_target = nactions

        action_pred = self.model(states)

        loss = F.mse_loss(action_pred, action_target, reduction='none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        nobs = self.normalizer.normalize(obs_dict)
        states = nobs['states']
        states = einops.rearrange(states, 'b h d -> b (h d)')
        action_pred = self.model(states)
        action_pred = self.normalizer['action'].unnormalize(action_pred)

        return {'action': action_pred[..., :2, :],
                'action_pred': action_pred[..., 2:, :]}
