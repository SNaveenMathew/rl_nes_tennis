import unittest
import torch
from tennis_rl.model import RecurrentActorCritic

class ModelTests(unittest.TestCase):
    def test_shapes_and_factorization(self):
        model = RecurrentActorCritic(4, 27, [32, 64, 64], 256, 256)
        obs = torch.zeros(4, 2, 4, 84, 84)
        hidden = model.initial_hidden(2, torch.device("cpu"))
        dist, values, next_hidden = model.forward_sequence(obs, hidden)
        self.assertEqual(dist.movement.logits.shape, (4, 2, 9))
        self.assertEqual(dist.shot.logits.shape, (4, 2, 3))
        self.assertEqual(values.shape, (4, 2))
        self.assertEqual(next_hidden.shape, (1, 2, 256))
        a = dist.sample()
        self.assertEqual(a.shape, (4, 2))
        self.assertTrue(torch.all((a >= 0) & (a < 27)))

    def test_act_returns_one_action_per_environment(self):
        model = RecurrentActorCritic(4, 27, [32, 64, 64], 256, 256)
        obs = torch.zeros(8, 4, 84, 84)
        hidden = model.initial_hidden(8, torch.device("cpu"))
        actions, logp, values, next_hidden = model.act(obs, hidden, deterministic=False)
        self.assertEqual(actions.shape, (8,))
        self.assertEqual(logp.shape, (8,))
        self.assertEqual(values.shape, (8,))
        self.assertEqual(next_hidden.shape, (1, 8, 256))
        self.assertTrue(torch.all((actions >= 0) & (actions < 27)))

if __name__ == "__main__":
    unittest.main()


def test_auxiliary_point_outcome_head():
    import torch
    from tennis_rl.model import RecurrentActorCritic
    model = RecurrentActorCritic(4, 27, [32,64,64], 64, 64)
    obs = torch.zeros(5, 2, 4, 84, 84)
    h = model.initial_hidden(2, torch.device("cpu"))
    dist, values, hn, outcome = model.forward_sequence(obs, h, include_aux=True)
    assert values.shape == (5,2)
    assert outcome.shape == (5,2)
    assert hn.shape == (1,2,64)


def test_motion_target_and_auxiliary_head():
    import torch
    from tennis_rl.model import RecurrentActorCritic, compute_spatial_motion_target
    model = RecurrentActorCritic(4, 27, [32, 64, 64], 64, 64)
    obs = torch.zeros(5, 2, 4, 84, 84)
    obs[:, :, 3, 40:44, 40:44] = 1.0  # simulate moving sprite
    targ = compute_spatial_motion_target(obs)
    assert targ.shape == (5, 2, 2)
    assert 0.0 <= targ.min() and targ.max() <= 1.0

    h = model.initial_hidden(2, torch.device("cpu"))
    dist, values, hn, motion_pred = model.forward_sequence(obs, h, include_motion=True)
    assert values.shape == (5, 2)
    assert motion_pred.shape == (5, 2, 2)
    assert 0.0 <= motion_pred.min() and motion_pred.max() <= 1.0

    # Test all heads combined
    dist, values, hn, outcome, motion_pred = model.forward_sequence(obs, h, include_aux=True, include_motion=True)
    assert outcome.shape == (5, 2)
    assert motion_pred.shape == (5, 2, 2)
