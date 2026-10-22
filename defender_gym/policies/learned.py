import numpy as np
import torch
import torch.nn as nn

from .base import Policy


class EntityPolicyNet(nn.Module):
    def __init__(self, n_features, n_kinds, d=64, heads=4):
        super().__init__()
        self.n_kinds = n_kinds
        self.encode = nn.Sequential(nn.Linear(n_features, d), nn.ReLU(), nn.Linear(d, d))
        self.attn = nn.TransformerEncoderLayer(d, heads, dim_feedforward=2 * d, dropout=0.0,
                                               batch_first=True)
        self.actor = nn.Linear(d, n_kinds)
        self.critic = nn.Linear(d, 1)

    def forward(self, obs, mask):
        x = self.attn(self.encode(obs), src_key_padding_mask=~mask)
        logits = self.actor(x).masked_fill(~mask[..., None], -1e9)
        m = mask[..., None].float()
        pooled = (x * m).sum(1) / m.sum(1).clamp(min=1.0)
        return logits.flatten(1), self.critic(pooled).squeeze(-1)


class LearnedPolicy(Policy):
    def __init__(self, n_features, n_kinds, d=64, heads=4):
        self.config = dict(n_features=n_features, n_kinds=n_kinds, d=d, heads=heads)
        self.net = EntityPolicyNet(n_features, n_kinds, d, heads)

    @classmethod
    def for_env(cls, env, **kwargs):
        return cls(env.observation_space.shape[1], int(env.action_space.nvec[1]), **kwargs)

    def distribution(self, obs, mask):
        logits, value = self.net(obs, mask)
        return torch.distributions.Categorical(logits=logits), value

    @torch.no_grad()
    def act(self, obs, mask, deterministic=False):
        obs_t = torch.as_tensor(obs, dtype=torch.float32)[None]
        mask_t = torch.as_tensor(mask, dtype=torch.bool)[None]
        dist, _ = self.distribution(obs_t, mask_t)
        idx = dist.probs.argmax(-1) if deterministic else dist.sample()
        return np.array(divmod(int(idx), self.net.n_kinds))

    def evaluate(self, obs, mask, actions):
        dist, value = self.distribution(obs, mask)
        flat = actions[:, 0].long() * self.net.n_kinds + actions[:, 1].long()
        return dist.log_prob(flat), dist.entropy(), value

    def save(self, path):
        torch.save({"config": self.config, "state": self.net.state_dict()}, path)

    def load(self, path):
        self.net.load_state_dict(torch.load(path)["state"])

    @classmethod
    def from_file(cls, path):
        ckpt = torch.load(path)
        policy = cls(**ckpt["config"])
        policy.net.load_state_dict(ckpt["state"])
        return policy
