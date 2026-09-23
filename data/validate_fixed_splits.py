"""Read-only validation of prepared training/dev/test splits (CPU standard library)."""
import argparse,json
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument('--data',required=True);a=p.parse_args();root=Path(a.data)
 train=json.loads((root/'train.json').read_text());keys={str(r['source_id']) for r in train}
 for file in root.glob('*.json'):
  if file.stem not in ['train','valid','initial_test','cl_test','official_test']:continue
  rows=json.loads(file.read_text());assert rows
  for r in rows:assert 0<r['prompt_len']<len(r['input_ids']) and r['output']
  if file.stem!='train':assert not keys & {str(r['source_id']) for r in rows},file
  print(file.stem,len(rows))
if __name__=='__main__':main()
