from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x + self.body(x))


class VisualEncoder(nn.Module):
    """Motion-aware Atari-style CNN.

    The four stacked grayscale frames remain the public observation contract.
    Internally we append frame differences so the representation contains an
    explicit short-horizon motion signal before the recurrent state is formed.
    """

    def __init__(
        self,
        in_channels: int,
        channels: list[int],
        embedding_dim: int,
        input_size: tuple[int, int] = (120, 128),
    ):
        super().__init__()
        if in_channels < 2:
            raise ValueError("VisualEncoder requires at least two stacked frames")
        c0, c1, c2 = channels[:3]
        internal_channels = in_channels + (in_channels - 1)
        self.stem = nn.Sequential(
            nn.Conv2d(internal_channels, c0, 8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(c0, c1, 4, stride=2),
            nn.ReLU(inplace=True),
        )
        self.residual = nn.Sequential(
            nn.Conv2d(c1, c2, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(c2),
            ResidualBlock(c2),
        )
        # Global average pooling used to sit here. It discards *where* features
        # are, and in tennis the ball/player positions are the whole game, so
        # the spatial feature map is flattened into the projection instead (the
        # standard Atari-DQN head). The flattened size depends on the frame
        # size, so measure it with a dummy pass.
        with torch.no_grad():
            probe = torch.zeros(1, internal_channels, *input_size)
            feature_map = self.residual(self.stem(probe))
        self.feature_shape = tuple(feature_map.shape[1:])  # (C, H', W')
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_map.numel(), embedding_dim),
            nn.ReLU(inplace=True),
        )
        self._init()

    def _init(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain("relu"))
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain("relu"))
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is [B, C, H, W], uint8/float in [0,255]/[0,1].
        if not torch.is_floating_point(x):
            x = x.float() / 255.0
        elif x.numel() and x.max().detach() > 1.5:
            x = x / 255.0
        diffs = x[:, 1:] - x[:, :-1]
        motion = torch.cat((x, diffs), dim=1)
        z = self.stem(motion)
        z = self.residual(z)
        return self.projection(z)


class JointCategorical:
    """A true categorical distribution over all 27 movement/button combinations.

    Movement and shot logits are retained as interpretable factor terms, but an
    additional pair-interaction logit makes the final joint action distribution
    expressive enough to learn that the best shot depends on movement (and vice
    versa). The action likelihood and entropy are calculated on the joint space.
    """

    def __init__(
        self,
        movement_logits: torch.Tensor,
        shot_logits: torch.Tensor,
        interaction_logits: torch.Tensor | None = None,
    ):
        self.movement = Categorical(logits=movement_logits)
        self.shot = Categorical(logits=shot_logits)
        joint = movement_logits.unsqueeze(-1) + shot_logits.unsqueeze(-2)
        if interaction_logits is not None:
            expected = (*joint.shape[:-2], 27)
            if tuple(interaction_logits.shape) != expected:
                raise ValueError(
                    f"interaction logits must have shape {expected}, got {tuple(interaction_logits.shape)}"
                )
            joint = joint + interaction_logits.reshape(*joint.shape[:-2], 9, 3)
        self.logits = joint.reshape(*joint.shape[:-2], 27)
        self.joint = Categorical(logits=self.logits)

    def sample(self) -> torch.Tensor:
        return self.joint.sample()

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        return self.joint.log_prob(action.long())

    def entropy(self) -> torch.Tensor:
        return self.joint.entropy()


def compute_spatial_motion_target(obs: torch.Tensor) -> torch.Tensor:
    """Computes the spatial center of mass (x, y) in [0, 1] of inter-frame motion.

    Court lines, net, and background in NES Tennis are static; temporal frame
    differences isolate moving entities (ball and moving player sprites).
    Computing the spatial centroid provides a self-supervised spatial target
    that grounds the visual representation in ball/player tracking.

    Args:
        obs: Tensor of shape (..., C, H, W) where C >= 2 stacked frames.
    Returns:
        Tensor of shape (..., 2) with coordinates [x_norm, y_norm] in [0, 1].
    """
    batch_shape = obs.shape[:-3]
    x = obs.reshape(-1, *obs.shape[-3:]).float()
    if x.max() > 1.5:
        x = x / 255.0
    diffs = torch.abs(x[:, 1:] - x[:, :-1])
    motion_map = diffs.sum(dim=1)
    # Threshold minor background dithering/noise
    motion_map = torch.where(motion_map > (4.0 / 255.0), motion_map, torch.zeros_like(motion_map))
    h, w = motion_map.shape[-2:]
    y_coords = torch.linspace(0.0, 1.0, h, device=obs.device, dtype=torch.float32).view(1, h, 1)
    x_coords = torch.linspace(0.0, 1.0, w, device=obs.device, dtype=torch.float32).view(1, 1, w)
    weights = motion_map.sum(dim=(-2, -1), keepdim=True)
    valid_mask = weights > 1e-4
    denom = torch.clamp(weights, min=1e-4)
    x_center = (motion_map * x_coords).sum(dim=(-2, -1), keepdim=True) / denom
    y_center = (motion_map * y_coords).sum(dim=(-2, -1), keepdim=True) / denom
    x_center = torch.where(valid_mask, x_center, torch.full_like(x_center, 0.5))
    y_center = torch.where(valid_mask, y_center, torch.full_like(y_center, 0.5))
    centroid = torch.cat([x_center.squeeze(-1), y_center.squeeze(-1)], dim=-1)
    return centroid.reshape(*batch_shape, 2)


class RecurrentActorCritic(nn.Module):
    """Motion-aware CNN + GRU actor-critic with factorized NES control policy."""

    def __init__(
        self,
        observation_channels: int,
        action_count: int,
        channels: list[int],
        embedding_dim: int,
        gru_hidden: int,
        gru_layers: int = 1,
        input_size: tuple[int, int] = (120, 128),
    ):
        super().__init__()
        if action_count != 27:
            raise ValueError("This controller model expects the 27-action NES Tennis control space")
        self.encoder = VisualEncoder(observation_channels, channels, embedding_dim, input_size)
        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=gru_hidden, num_layers=gru_layers)
        self.movement_head = nn.Linear(gru_hidden, 9)
        self.shot_head = nn.Linear(gru_hidden, 3)
        # Zero-init interaction keeps the initial distribution equivalent to the
        # previous factorized policy while allowing training to learn movement x
        # shot dependencies instead of assuming conditional independence.
        self.joint_interaction_head = nn.Linear(gru_hidden, 27)
        self.value_head = nn.Linear(gru_hidden, 1)
        # Auxiliary point-outcome head: predicts whether the current partial
        # point trajectory ultimately wins the point. This is trained only from
        # recorded point outcomes; it does not turn every action in a losing
        # point into a "bad" action label.
        self.point_outcome_head = nn.Linear(gru_hidden, 1)
        # Auxiliary spatial motion head: predicts the normalized 2D coordinate [x, y]
        # in [0, 1] of the dominant moving entity (ball/player) via self-supervision.
        self.motion_head = nn.Linear(gru_hidden, 2)
        self.gru_hidden = gru_hidden
        self.gru_layers = gru_layers
        nn.init.orthogonal_(self.movement_head.weight, gain=0.01)
        nn.init.zeros_(self.movement_head.bias)
        nn.init.orthogonal_(self.shot_head.weight, gain=0.01)
        nn.init.zeros_(self.shot_head.bias)
        nn.init.zeros_(self.joint_interaction_head.weight)
        nn.init.zeros_(self.joint_interaction_head.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)
        nn.init.orthogonal_(self.point_outcome_head.weight, gain=0.01)
        nn.init.zeros_(self.point_outcome_head.bias)
        nn.init.orthogonal_(self.motion_head.weight, gain=0.01)
        nn.init.zeros_(self.motion_head.bias)

    def initial_hidden(self, batch: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(self.gru_layers, batch, self.gru_hidden, device=device)

    def forward_sequence(
        self,
        obs: torch.Tensor,
        hidden: torch.Tensor,
        *,
        include_aux: bool = False,
        include_motion: bool = False,
    ):
        t, b = obs.shape[:2]
        x = obs.reshape(t * b, *obs.shape[2:])
        embedding = self.encoder(x).reshape(t, b, -1)
        recurrent, next_hidden = self.gru(embedding, hidden)
        move_logits = self.movement_head(recurrent)
        shot_logits = self.shot_head(recurrent)
        interaction_logits = self.joint_interaction_head(recurrent)
        values = self.value_head(recurrent).squeeze(-1)
        dist = JointCategorical(move_logits, shot_logits, interaction_logits)

        point_outcome_logits = None
        if include_aux:
            point_outcome_logits = self.point_outcome_head(recurrent).squeeze(-1)

        motion_coords = None
        if include_motion:
            motion_coords = torch.sigmoid(self.motion_head(recurrent))

        if include_aux and include_motion:
            return dist, values, next_hidden, point_outcome_logits, motion_coords
        if include_aux:
            return dist, values, next_hidden, point_outcome_logits
        if include_motion:
            return dist, values, next_hidden, motion_coords
        return dist, values, next_hidden

    @torch.no_grad()
    def act(self, obs: torch.Tensor, hidden: torch.Tensor, deterministic: bool = False):
        dist, values, next_hidden = self.forward_sequence(obs.unsqueeze(0), hidden)
        if deterministic:
            action = torch.argmax(dist.logits[0], dim=-1)
        else:
            action = dist.sample()[0]
        log_prob = dist.log_prob(action.unsqueeze(0))[0]
        entropy = dist.entropy()[0]
        return action, log_prob, values[0], next_hidden
