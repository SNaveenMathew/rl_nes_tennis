from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical


class VisualEncoder(nn.Module):
    def __init__(self, in_channels: int, channels: list[int], embedding_dim: int):
        super().__init__()
        layers: list[nn.Module] = []
        c = in_channels
        for out_c, kernel, stride in zip(channels, (8, 4, 3), (4, 2, 1)):
            layers.extend([
                nn.Conv2d(c, out_c, kernel, stride),
                nn.ReLU(inplace=True),
            ])
            c = out_c
        layers.extend([
            nn.Flatten(),
            nn.LazyLinear(embedding_dim),
            nn.ReLU(inplace=True),
        ])
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RecurrentActorCritic(nn.Module):
    def __init__(
        self,
        observation_channels: int,
        action_count: int,
        channels: list[int],
        embedding_dim: int,
        gru_hidden: int,
        gru_layers: int = 1,
    ):
        super().__init__()
        self.encoder = VisualEncoder(observation_channels, channels, embedding_dim)
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
        )
        self.policy_head = nn.Linear(gru_hidden, action_count)
        self.value_head = nn.Linear(gru_hidden, 1)
        self.gru_hidden = gru_hidden
        self.gru_layers = gru_layers

        # MLP orthogonal initialization is standard for PPO actor/critic heads.
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.zeros_(self.policy_head.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def initial_hidden(self, batch: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(self.gru_layers, batch, self.gru_hidden, device=device)

    def forward_sequence(
        self,
        obs: torch.Tensor,
        hidden: torch.Tensor,
    ) -> tuple[Categorical, torch.Tensor, torch.Tensor]:
        """obs: [T, B, C, H, W], hidden: [L, B, H]."""
        t, b = obs.shape[:2]
        x = obs.reshape(t * b, *obs.shape[2:])
        embedding = self.encoder(x).reshape(t, b, -1)
        recurrent, next_hidden = self.gru(embedding, hidden)
        logits = self.policy_head(recurrent)
        values = self.value_head(recurrent).squeeze(-1)
        return Categorical(logits=logits), values, next_hidden

    @torch.no_grad()
    def act(
        self,
        obs: torch.Tensor,
        hidden: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, values, next_hidden = self.forward_sequence(obs.unsqueeze(0), hidden)
        dist = Categorical(logits=dist.logits[0])
        if deterministic:
            action = torch.argmax(dist.logits, dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, values[0], next_hidden
