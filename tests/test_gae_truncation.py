import torch
from tennis_rl.ppo import compute_gae_with_truncation


def test_truncated_transition_bootstraps_final_observation_not_reset_state():
    # t=0 is truncated; values[1] intentionally models a new episode reset state.
    rewards = torch.tensor([[0.0], [0.0]])
    values = torch.tensor([[1.0], [100.0]])
    dones = torch.tensor([[True], [False]])
    truncated = torch.tensor([[True], [False]])
    truncated_values = torch.tensor([[7.0], [0.0]])
    last_values = torch.tensor([3.0])
    adv, returns = compute_gae_with_truncation(
        rewards, values, dones, truncated, truncated_values, last_values,
        gamma=0.9, gae_lambda=0.95,
    )
    assert torch.isclose(returns[0, 0], torch.tensor(0.9 * 7.0))
    assert not torch.isclose(returns[0, 0], torch.tensor(0.9 * 100.0))


def test_true_terminal_does_not_bootstrap():
    rewards = torch.tensor([[1.0]])
    values = torch.tensor([[0.5]])
    dones = torch.tensor([[True]])
    truncated = torch.tensor([[False]])
    truncated_values = torch.tensor([[99.0]])
    last_values = torch.tensor([100.0])
    adv, returns = compute_gae_with_truncation(
        rewards, values, dones, truncated, truncated_values, last_values, 0.99, 0.95
    )
    assert torch.isclose(returns[0, 0], torch.tensor(1.0))
