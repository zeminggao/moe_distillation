import os
import json, hashlib, random, concurrent.futures
from pathlib import Path
import requests
import pyarrow as pa

ROOT=Path(os.environ['CITB_WORK_DIR']).resolve()
RAW=Path(os.environ['CITB_SOURCE_DIR']).resolve()
tree={x['path']:x for x in json.loads((Path(__file__).parent/'source_files.json').read_text())}
(ROOT/'data').mkdir(parents=True,exist_ok=True)
def download(path):
    dest=RAW/path
    data=dest.read_bytes()
    assert hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()==tree[path]['sha'], 'Upstream file differs from recorded split source: '+path
    return dest
prefix='data/CIT_data/initial_multitask_learning/defintion_pos_2/'
data={}
for split in ['train','dev','test']:
    path=prefix+split+'/data-00000-of-00001.arrow'
    f=download(path)
    data['initial_'+split]=pa.ipc.open_stream(f).read_all().to_pylist()
    print(split,len(data['initial_'+split]),flush=True)
f=download('data/CIT_data/official_test_data/data-00000-of-00001.arrow')
data['official_test']=pa.ipc.open_stream(f).read_all().to_pylist()
random.seed(42)
for s in ['train','dev','test']:data['cl_'+s]=[]
for task in download('data/splits/CIT_splits/cl_38_random_tasks.txt').read_text().split():
    d=json.loads(download('data/tasks/'+task+'.json').read_text(encoding='utf-8'))
    instances=d.pop('Instances');d['Task']=task;d.pop('Instruction Source',None)
    rows=[dict(d,id=i['id'],Instance=i) for i in instances]
    data['cl_test']+=rows[:25];data['cl_dev']+=rows[25:50]
    remaining=rows[50:]
    if len(remaining)>=100:random.shuffle(remaining)
    data['cl_train']+=remaining[:100]
data['train']=data.pop('initial_train')+data.pop('cl_train')
data['dev']=data.pop('initial_dev')+data.pop('cl_dev')
manifest={'source':'CITB released initial Arrow datasets + long-stream MULTI_TASK 100/25/25 CL split','seed':42,'counts':{},'sha256':{},'tasks':{},'deduplication':'none, matching official concatenate_datasets'}
for name,rows in data.items():
    f=ROOT/'data'/f'{name}.json';f.write_text(json.dumps(rows,ensure_ascii=False),encoding='utf-8')
    manifest['counts'][name]=len(rows);manifest['sha256'][name]=hashlib.sha256(f.read_bytes()).hexdigest()
    manifest['tasks'][name]=len({r['Task'] for r in rows})
assert manifest['counts']['train']==13411
(ROOT/'data/manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(manifest,indent=2),flush=True)
