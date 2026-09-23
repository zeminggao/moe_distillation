"""Paper Eq. 7--9. Independent from the running historical-KID implementation."""
import types
import torch
from torch.utils.checkpoint import checkpoint


def ka_indices(logits, probability=.05, generator=None):
    """Independent token/layer mixture; weighted sampling WITHOUT replacement."""
    k = logits.shape[-1] - 1
    top = logits.topk(k, dim=-1).indices
    if probability == 0:
        return top
    sampled = torch.multinomial(logits.float().softmax(-1), k,
                                replacement=False, generator=generator)
    choose = torch.rand((len(logits), 1), device=logits.device,
                        generator=generator) < probability
    return torch.where(choose, sampled, top)


def cv_squared(x):
    return x.float().var() / (x.float().mean().square() + 1e-10)


def sequence_kl(first, second, mask):
    """Mean per-response KL(first || second), chunked over selected tokens."""
    counts = mask.sum(-1)
    if not bool((counts > 0).all()):
        raise ValueError('Empty response')
    weights = (1 / counts.float())[:, None].expand_as(mask)[mask] / len(mask)
    a, b = first[mask], second[mask]
    loss = a.new_zeros((), dtype=torch.float32)
    def term(x, y, w):
        lp, lq = x.float().log_softmax(-1), y.float().log_softmax(-1)
        return ((lp.exp() * (lp - lq)).sum(-1) * w).sum()
    for start in range(0, len(a), 128):
        loss = loss + checkpoint(term, a[start:start+128], b[start:start+128],
                                 weights[start:start+128], use_reentrant=False)
    return loss


class ExpertMethod:
    def __init__(self, teacher, method, sampling_probability=.05):
        if method not in ('ka', 'sar', 'top8', 'top2'):
            raise ValueError(method)
        self.teacher, self.method = teacher, method
        self.sampling_probability = sampling_probability
        self.collect = False
        self.sums = None
        self.balance_coefficients = None
        self.attention_mask = None
        teacher.eval().requires_grad_(False)
        self.router_parameters = []
        if method == "top2":return  # Preserve the exact native gate implementation.
        for layer_id, layer in enumerate(teacher.model.layers):
            gate = layer.mlp.gate
            assert gate.num_experts == 8 and gate.num_selects == 2 and gate.use_softmax
            if method == 'sar':
                for module in (gate.gate_network, gate.weight_noise):
                    module.float().requires_grad_(True)
                    self.router_parameters.extend(module.parameters())
                # Non-reentrant checkpointing works even with frozen embeddings.
                old_layer = layer.forward
                def layer_forward(m, *args, old=old_layer, **kwargs):
                    if torch.is_grad_enabled():
                        return checkpoint(old, *args, use_reentrant=False, **kwargs)
                    return old(*args, **kwargs)
                layer.forward = types.MethodType(layer_forward, layer)

            def forward(g, x, index=layer_id):
                logits = g.gate_network(x)
                if self.method == 'sar':
                    scale = g.softplus(g.weight_noise(x)) + g.noise_epsilon
                    logits = logits + torch.randn_like(logits) * scale
                    indices = torch.arange(8, device=x.device).expand(len(x), 8)
                elif self.method == 'top8':
                    indices = logits.topk(8, dim=-1).indices
                elif self.method == 'top2':
                    indices = logits.topk(3, dim=-1).indices[:,:2]
                else:
                    indices = ka_indices(logits, self.sampling_probability)
                scores = logits.gather(1, indices).float().softmax(-1).to(x.dtype)
                full = torch.zeros_like(logits).scatter(1, indices, scores.to(logits.dtype))
                valid = self.attention_mask.reshape(-1).to(full.dtype)[:, None]
                importance = (full.float() * valid).sum(0)
                load = ((full > 0) * valid.bool()).sum(0)
                if self.collect:
                    self.sums[index].add_(importance.detach())
                # Global-batch CV derivative supplied by the replay prepass.
                balance = logits.new_zeros((), dtype=torch.float32)
                if self.balance_coefficients is not None:
                    balance = (importance * self.balance_coefficients[index]).sum()
                return dict(topK_indices=indices, topK_scores=scores,
                            balance_loss=balance, load=load, importance=importance)
            gate.forward = types.MethodType(forward, gate)

    def set_batch(self, attention_mask):
        self.attention_mask = attention_mask

    def router_state(self):
        return {name: value.detach().cpu() for name, value in self.teacher.named_parameters()
                if value.requires_grad}

    def load_router_state(self, state):
        expected = {n: p for n, p in self.teacher.named_parameters() if p.requires_grad}
        if expected.keys() != state.keys():
            raise ValueError('Router state keys differ')
        with torch.no_grad():
            for name, parameter in expected.items():
                parameter.copy_(state[name])
