"""Historical response-target expansion, with original six logical rank budgets."""
import random
import types
import torch
import numpy as np
from dynamic_experts import DynamicExperts
from progress_controller import NativeKLProgress

def logical_rows(rows, rank):
    assert len(rows) == 384 and rank in (0, 1)
    return [row for logical in range(rank, 6, 2) for row in rows[logical::6]]

def response_mask(rows, length, device):
    mask = torch.zeros((len(rows), length-1), dtype=torch.bool, device=device)
    for j, row in enumerate(rows):
        mask[j, row['prompt_len']-1:len(row['input_ids'])-1] = True
    return mask

def chunk_attention(teacher):
    # Only attention is split. MoE still observes all 64 logical-rank samples.
    for layer in teacher.model.layers:
        old = layer.self_attn.forward
        def forward(module, hidden_states, attention_mask=None, position_ids=None,
                    past_key_value=None, output_attentions=False, use_cache=False, _old=old, **kw):
            assert not use_cache and past_key_value is None and not output_attentions
            result = []
            for start in range(0, len(hidden_states), 4):
                def part(x):
                    return x[start:start+4] if x is not None and x.shape[0] == len(hidden_states) else x
                y = _old(hidden_states[start:start+4], attention_mask=part(attention_mask),
                         position_ids=part(position_ids), past_key_value=None,
                         output_attentions=False, use_cache=False, **kw)
                result.append(y[0])
            return torch.cat(result), None, None
        layer.self_attn.forward = types.MethodType(forward, layer.self_attn)

class HistoricalPolicy:
    def __init__(self, teacher, path, special_ids, kid=False):
        self.teacher = teacher
        self.control = DynamicExperts(teacher)
        self.table = np.load(path)
        assert self.table.dtype == np.uint8 and self.table.ndim == 2
        self.special_ids = special_ids
        self.router_parameters = []
        self.progress = NativeKLProgress(interval=10, warmup_windows=4, beta=.9, alpha=.5, cutoff=.05) if kid else None
        if kid: chunk_attention(teacher)

    def set_tokens(self, ids, mask):
        targets = ids[:, 1:].detach().cpu().numpy()
        bits = torch.zeros((*ids.shape, self.table.shape[1]), dtype=torch.uint8, device=ids.device)
        active = mask.clone()
        for token in self.special_ids: active &= ids[:, 1:] != token
        bits[:, :-1] = torch.from_numpy(self.table[targets]).to(ids.device) * active[..., None]
        self.control.set_batch(bits, active)

    def clear(self):
        self.control.bits = None
        self.control.response_mask = None

    def state_dict(self):
        return self.progress.state_dict() if self.progress else None

    def load_state_dict(self, state):
        if self.progress:
            assert state is not None
            self.progress.load_state_dict(state)
            self.control.multiplier = self.progress.multiplier

class FixedProbe:
    def __init__(self, teacher, policy, data, special_ids, rank, world):
        positions = [(i,k-1) for i,r in enumerate(data) for k in range(r['prompt_len'],len(r['input_ids']))
                     if r['input_ids'][k] not in special_ids]
        selected = random.Random(100045).sample(positions,4096)
        self.selected = selected
        groups = {}
        for i,p in selected[rank::world]:groups.setdefault(i,[]).append(p)
        self.entries = []
        policy.control.enabled = False
        with torch.no_grad(), torch.autocast('cuda',dtype=torch.bfloat16):
            for i, positions in groups.items():
                ids = torch.tensor([data[i]['input_ids']],device='cuda')
                q = teacher(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False).logits[0,positions]
                self.entries.append((ids.cpu(),positions,q.cpu()))
        policy.control.enabled = True

    def score(self, student):
        value = torch.zeros(2,device='cuda',dtype=torch.float64)
        with torch.no_grad(), torch.autocast('cuda',dtype=torch.bfloat16):
            for cpu_ids, positions, q in self.entries:
                ids=cpu_ids.cuda()
                p=student(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False).logits[0,positions].float()
                value[0] += torch.nn.functional.kl_div(p.log_softmax(-1),q.cuda().float().softmax(-1),reduction='sum')
                value[1] += len(positions)
        return value
