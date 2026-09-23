"""CPU semantic checks; no production model or GPU allocation."""
import copy
import unittest
import torch
from history_policy import logical_rows
from dynamic_experts import retain_extras
from progress_controller import NativeKLProgress
from common import schedule, normalize

class Semantics(unittest.TestCase):
    def test_duplicate_ids_preserve_all_examples(self):
        data=[dict(source_id='same',input_ids=[1,3,4],prompt_len=1,output='a'),
              dict(source_id='same',input_ids=[1,5,6],prompt_len=1,output='b')]
        rows,_=normalize(data,2,3,2)
        self.assertEqual([r['input_ids'] for r in rows],[r['input_ids'] for r in data])
        self.assertEqual(len({r['source_id'] for r in rows}),2)
        self.assertEqual([r['original_source_id'] for r in rows],['same','same'])
    def test_logical_rank_partition(self):
        original=list(range(384))
        physical=[logical_rows(original,r) for r in (0,1)]
        self.assertEqual(sorted(physical[0]+physical[1]),original)
        for rank in (0,1):
            for i,logical in enumerate(range(rank,6,2)):
                self.assertEqual(physical[rank][64*i:64*(i+1)],original[logical::6])

    def test_budget_matches_full_reference_and_stable_ties(self):
        torch.manual_seed(7)
        extra=torch.rand(64,11,8)>.6
        probability=torch.randint(0,4,extra.shape).float()/4
        for m in [0.,.05,.37,1.]:
            actual=retain_extras(extra,probability,m)
            candidates=[i for i in range(extra.numel()) if extra.flatten()[i]]
            candidates.sort(key=lambda i:(-float(probability.flatten()[i]),i))
            reference=torch.zeros(extra.numel(),dtype=torch.bool)
            reference[candidates[:int(len(candidates)*m)]]=True
            self.assertTrue(torch.equal(actual.flatten(),reference))

    def test_microbatch_independent_budget_is_detectably_different(self):
        extra=torch.ones(64,1,8,dtype=torch.bool)
        p=torch.arange(extra.numel()).reshape(extra.shape).float()
        full=retain_extras(extra,p,.5)
        split=torch.cat([retain_extras(extra[i:i+4],p[i:i+4],.5) for i in range(0,64,4)])
        self.assertFalse(torch.equal(full,split))

    def test_progress_exact_resume(self):
        p=NativeKLProgress(interval=10)
        for step,kl in [(0,2),(10,1.7),(20,1.4),(30,1.3),(40,1.2),(50,1.19)]:p.observe(step,kl)
        q=NativeKLProgress(interval=10);q.load_state_dict(copy.deepcopy(p.state_dict()))
        for step,kl in [(60,1.185),(70,1.184),(80,1.184)]:
            self.assertEqual(p.observe(step,kl),q.observe(step,kl))

    def test_counts_drop_last_and_save_count(self):
        for n,ep,expected in [(10949,5,140),(6000,5,75),(13411,10,340)]:
            per,saves=schedule(n,384,ep)
            self.assertEqual(per*ep,expected)
            self.assertEqual(len(set(saves)),2*ep)
            self.assertEqual(saves[-1],expected)

if __name__=='__main__':unittest.main()
