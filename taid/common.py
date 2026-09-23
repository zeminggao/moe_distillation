import hashlib
import json
import math
from pathlib import Path
import torch


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.partial')
    tmp.write_text(json.dumps(obj, indent=2), encoding='utf-8')
    tmp.replace(path)


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def schedule(n, batch, epochs):
    # Official data loader drops an incomplete training batch.
    steps = n // batch
    assert steps >= 2
    return steps, [e*steps+j for e in range(epochs) for j in (math.ceil(steps/2), steps)]


def normalize(rows, prompt_cap, response_cap, eos):
    """Retain legacy tokenization, truncate response independently; no fake EOS."""
    from collections import Counter
    counts=Counter(r['source_id'] for r in rows);occurrences=Counter()
    result, truncated = [], 0
    for row in rows:
        p = row['prompt_len']; ids = row['input_ids']
        assert 0 < p <= prompt_cap and len(ids) > p
        response = ids[p:]
        if len(response) > response_cap:
            response = response[:response_cap]; truncated += 1
        source=row['source_id'];occurrences[source]+=1
        record=dict(row,input_ids=ids[:p]+response)
        if counts[source]>1:
            record['original_source_id']=source
            record['source_id']=str(source)+'::instance'+str(occurrences[source])
        result.append(record)
    assert len({r['source_id'] for r in result}) == len(result)
    return result, truncated


def collate(rows, pad, device='cpu', stress=False, prompt_cap=256, response_cap=256):
    if stress:
        rows = [dict(r, prompt_len=prompt_cap,
                     input_ids=[r['input_ids'][0]]*prompt_cap +
                     [r['input_ids'][r['prompt_len']]]*(response_cap-1)+[pad]) for r in rows]
    length = max(len(r['input_ids']) for r in rows)
    ids = torch.full((len(rows), length), pad, dtype=torch.long, device=device)
    attention = torch.zeros_like(ids)
    mask = torch.zeros((len(rows), length-1), dtype=torch.bool, device=device)
    for j,r in enumerate(rows):
        n = len(r['input_ids']); ids[j,:n] = torch.tensor(r['input_ids'],device=device)
        attention[j,:n] = 1
        # Official prepare_ultrachat.py sets labels=input_ids for the whole text.
        mask[j,:n-1] = True
    return ids, attention, mask


def parameter_groups(model):
    # Exact official exclusion: nn.LayerNorm and any parameter name with 'bias'.
    def names(module):
        output = []
        for name, child in module.named_children():
            if not isinstance(child, torch.nn.LayerNorm):
                output += [name+'.'+n for n in names(child)]
        return output + list(module._parameters)
    decay = {n for n in names(model) if 'bias' not in n}
    return [dict(params=[p for n,p in model.named_parameters() if n in decay and p.requires_grad]),
            dict(params=[p for n,p in model.named_parameters() if n not in decay and p.requires_grad],weight_decay=0.)]


def verify_checkpoint(path):
    path=Path(path); manifest=read(path/'.ready.json')
    for name,h in manifest.items():
        assert sha(path/name)==h, f'Checkpoint SHA mismatch: {name}'
    return manifest
