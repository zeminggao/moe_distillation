"""TAID objective/state, adapted from the pinned SakanaAI reference.

The reference forward_kl returns soft-target cross entropy (without target
entropy). Preserve that scalar for the adaptive controller as well as gradients.
One controller update per global optimizer step; validation is read-only.
"""
from dataclasses import dataclass, asdict
import math
import torch
import torch.nn.functional as F


def loss_sum(student, teacher, mask, t):
    target = F.softmax((1 - t) * student.detach() + t * teacher.detach(),
                       dim=-1, dtype=torch.float32)
    logp = F.log_softmax(student, dim=-1, dtype=torch.float32)
    product = (target * logp).masked_fill(torch.isinf(student), 0)
    return -(product.sum(-1) * mask).sum()


@dataclass
class Controller:
    t_start: float = .2
    t_end: float = 1.
    alpha: float = 5e-4
    beta: float = .99
    t: float = .2
    prev_loss: float | None = None
    momentum: float = 0.

    def __post_init__(self):
        assert 0 <= self.t_start < self.t_end <= 1
        assert 0 <= self.alpha <= 1 and 0 <= self.beta < 1

    def update(self, loss, global_step, total_steps):
        assert math.isfinite(loss) and loss >= 0
        if self.prev_loss is None:
            self.prev_loss = loss
            return
        relative = (self.prev_loss - loss) / (self.prev_loss + 1e-15)
        self.momentum = self.beta * self.momentum + (1 - self.beta) * relative
        sigmoid = 1 / (1 + math.exp(-max(-700., min(700., self.momentum))))
        linear = self.t_start + (self.t_end - self.t_start) * global_step / total_steps
        self.t = min(self.t_end, max(linear, self.t + self.alpha * sigmoid * (1-self.t)))
        self.prev_loss = loss

    def state_dict(self):
        return asdict(self)
