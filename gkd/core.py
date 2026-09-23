import hashlib, json, math
from pathlib import Path

def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path, obj):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_name(path.name+'.partial'); tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8'); tmp.replace(path)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()
def model_fingerprint(path):
    root=Path(path)
    files=sorted(set(root.glob('*.safetensors')) | set(root.glob('pytorch_model*.bin')))
    if not files:raise FileNotFoundError(f'No model weights in {root}')
    files+=sorted(root.glob('config.json'))
    return {f.name:sha(f) for f in files}
def save_steps(n,batch,epochs):
    per=math.ceil(n/batch)
    if per<2: raise ValueError('Need at least two batches per epoch for 20 checkpoints')
    return sorted({e*per+j for e in range(epochs) for j in (math.ceil(per/2),per)})
def rank_indices(order,start,batch,rank,world): return order[start:start+batch][rank::world]
def weighted_scale(micro_count,global_count,world): return world*micro_count/global_count
