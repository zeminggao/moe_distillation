"""Packaging checks and public configuration CLI tests; no torch or GPU required."""
import ast,importlib.util,json,subprocess,sys,tempfile,unittest
from pathlib import Path
R=Path(__file__).resolve().parents[1]
class ReleaseTests(unittest.TestCase):
 def test_syntax_and_no_private_deploy_endpoints(self):
  for p in R.rglob('*.py'):
   text=p.read_text(encoding='utf-8');ast.parse(text,filename=str(p))
   if p==Path(__file__).resolve():continue
   for forbidden in ['bupt.cc','seetacloud.com','gaozeming','TAID_S13_PASSWORD','deita']:
    self.assertNotIn(forbidden,text.lower(),str(p))
 def test_configuration_all_settings(self):
  with tempfile.TemporaryDirectory() as tmp:
   base=Path(tmp);model=base/'model';model.mkdir();(model/'config.json').write_text('{}')
   data=base/'data';data.mkdir();rows=[dict(source_id='a',prompt_len=1,input_ids=[1,2],output=['x'])]
   for split in ['train','valid']:(data/(split+'.json')).write_text(json.dumps(rows))
   import hashlib
   history=data/'history';history.mkdir();(history/'m80_expert_bitmask.npy').write_bytes(b'test-only')
   (history/'manifest.json').write_text(json.dumps(dict(teacher=str(model.resolve()),train_sha256=hashlib.sha256((data/'train.json').read_bytes()).hexdigest())))
   for objective in ['gkd','taid']:
    for dataset in ['dolly','citb']:
     for method in ['top2','top8','strawman','kid','ka','sar']:
      out=base/f'{objective}_{dataset}_{method}'
      subprocess.run([sys.executable,str(R/'configure.py'),'--objective',objective,'--dataset',dataset,'--method',method,'--teacher',str(model),'--student',str(model),'--data',str(data),'--output',str(out)],check=True,stdout=subprocess.DEVNULL)
      c=json.loads((out/'config.json').read_text());self.assertEqual(c['global_batch'],384)
      self.assertEqual(c['world_size'],6 if objective=='gkd' else 2)
      self.assertEqual(c['epochs'],5 if objective=='taid' and dataset=='dolly' else 10)
 def test_gkd_tail_coverage_and_accumulation(self):
  spec=importlib.util.spec_from_file_location('release_core',R/'gkd/core.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
  for n in [10949,13411]:
   ids=[i for start in range(0,n,384) for rank in range(6) for i in m.rank_indices(list(range(n)),start,384,rank,6)]
   self.assertEqual(sorted(ids),list(range(n)));self.assertEqual(len(m.save_steps(n,384,10)),20)
  n=197;values=[i%17/17 for i in range(n)];result=0
  for rank in range(6):
   local=values[rank::6]
   for off in range(0,len(local),4):
    chunk=local[off:off+4];result+=sum(chunk)/len(chunk)*m.weighted_scale(len(chunk),n,6)/6
  self.assertAlmostEqual(result,sum(values)/n)
if __name__=='__main__':unittest.main()
