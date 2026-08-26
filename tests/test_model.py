import unittest

import torch

from tennis_rl.model import RecurrentActorCritic


class ModelTests(unittest.TestCase):
    def test_shapes(self):
        model = RecurrentActorCritic(4, 19, [32, 64, 64], 256, 256)
        obs = torch.zeros(4, 2, 4, 84, 84)
        hidden = model.initial_hidden(2, torch.device("cpu"))
        dist, values, next_hidden = model.forward_sequence(obs, hidden)
        self.assertEqual(dist.logits.shape, (4, 2, 19))
        self.assertEqual(values.shape, (4, 2))
        self.assertEqual(next_hidden.shape, (1, 2, 256))


if __name__ == "__main__":
    unittest.main()
