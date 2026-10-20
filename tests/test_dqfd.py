import numpy as np
import torch

from tennis_rl.config import PPOConfig
from tennis_rl.dqfd import DQFDTrainer, RecurrentDuelingQ, Sequence
from tennis_rl.model import RecurrentActorCritic


def _trainer():
    cfg = PPOConfig(sequence_length=8, sequence_batch_size=2, demo_replay_capacity=10,
                    online_replay_capacity=10, demo_fraction=0.5, demo_pretrain_updates=1,
                    updates_per_rollout=1, target_update_every=2)
    base = RecurrentActorCritic(4, 27, [16, 32, 32], 32, 32, input_size=(84, 84))
    model = RecurrentDuelingQ(base)
    import copy
    target = RecurrentDuelingQ(copy.deepcopy(base))
    return DQFDTrainer(model, target, cfg, torch.device('cpu'))


def test_sequence_replay_update():
    tr = _trainer()
    for demo in (True, False):
        tr.add_sequence(Sequence(
            obs=np.zeros((8,4,84,84), np.uint8),
            actions=np.arange(8,dtype=np.int64)%27,
            rewards=np.zeros(8,np.float32),
            dones=np.array([False]*7+[True]),
            demo=demo), priority=2.0)
    m = tr.update(2)
    assert m is not None
    assert m['loss'] >= 0


def test_demo_preference_available():
    tr=_trainer()
    seq=Sequence(np.zeros((8,4,84,84),np.uint8),np.zeros(8,np.int64),np.zeros(8,np.float32),np.ones(8,np.bool_),True)
    tr.add_sequence(seq, priority=1.0)
    assert len(tr.replay.demo)==1


def test_target_network_is_independent():
    tr = _trainer()
    a = next(tr.model.parameters())
    b = next(tr.target.parameters())
    assert a.data_ptr() != b.data_ptr()
