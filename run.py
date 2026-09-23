"""Explicit finite launcher; no SSH, automatic queue, shutdown, or GPU allocation."""
import argparse,json,subprocess,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def main():
 p=argparse.ArgumentParser();p.add_argument('--objective',choices=['gkd','taid'],required=True);p.add_argument('--config',required=True);p.add_argument('--stage',choices=['preflight','train'],required=True);p.add_argument('--resume');a=p.parse_args()
 path=Path(a.config).resolve();c=json.loads(path.read_text());assert c['dataset'] in ('dolly','citb');job=path.parent
 base=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node',str(c['world_size']),str(ROOT/a.objective/'train.py')]
 if a.objective=='taid':
  base+=['--config',str(path),'--mode',a.stage]
  if a.resume:base+=['--resume',str(Path(a.resume).resolve())]
  subprocess.run(base,check=True)
  if a.stage=='preflight':subprocess.run(base+['--stress'],check=True)
 else:
  import os
  base+=['--run-dir',str(job)]
  if a.stage=='preflight':
   subprocess.run(base+['--steps','3'],check=True);subprocess.run(base+['--steps','2','--stress'],check=True)
   (job/'preflight_passed.json').write_text(json.dumps({'config_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'code':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'gkd').glob('*.py')}}))
  else:
   pre=json.loads((job/'preflight_passed.json').read_text());assert pre['config_sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
   assert pre['code']=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'gkd').glob('*.py')},'Rerun preflight after code changes'
   env=dict(os.environ)
   if a.resume:env['RESUME_CHECKPOINT']=str(Path(a.resume).resolve())
   subprocess.run(base,env=env,check=True)
if __name__=='__main__':main()
