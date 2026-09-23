"""CPU semantic tests; no GPU training or large model loading."""
import copy
import math
from pathlib import Path
import unittest
import torch
torch.set_num_threads(2)
from common import collate, normalize, schedule, parameter_groups
from taid import Controller, loss_sum

ROOT=Path(__file__).parent


class Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_official_loss_and_gradient(self):
        # Execute pinned author's function, removing imports of project wrappers.
        text=(ROOT/'official/src__distil_losses__fkl.py').read_text()
        text=text.replace('from .base import DistilLoss','DistilLoss = object')
        ns={};exec(text,ns)
        for t in [.2,.57,1.]:
            x=torch.randn(3,7,19,requires_grad=True);y=torch.randn_like(x)
            m=torch.rand(3,7)>.3
            target=torch.softmax((1-t)*x.detach()+t*y,dim=-1,dtype=torch.float32)
            ref=ns['forward_kl'](x,y,m,teacher_probs=target)
            actual=loss_sum(x,y,m,t)/m.sum()
            torch.testing.assert_close(actual,ref)
            torch.testing.assert_close(torch.autograd.grad(actual,x,retain_graph=True)[0],torch.autograd.grad(ref,x)[0])

    def test_accumulation_and_two_rank_scaling(self):
        x=torch.randn(12,7,19,requires_grad=True);y=torch.randn_like(x);m=torch.rand(12,7)>.3
        full=loss_sum(x,y,m,.3)/m.sum();expected=torch.autograd.grad(full,x)[0]
        # Simulate two ranks, uneven valid-token counts, DDP's mean gradient.
        out=torch.zeros_like(x)
        for rank in range(2):
            ix=list(range(rank,12,2))
            for start in range(0,len(ix),2):
                ids=ix[start:start+2]
                v=loss_sum(x[ids],y[ids],m[ids],.3)*2/m.sum()
                out+=torch.autograd.grad(v,x)[0]/2
        torch.testing.assert_close(out,expected)

    def test_official_controller(self):
        text=(ROOT/'official/src__distil_losses__taid.py').read_text()
        text=text.replace('from lightning import LightningModule','LightningModule = object')
        text=text.replace('from .base import DistilLoss','DistilLoss = torch.nn.Module')
        text=text.replace('from .fkl import forward_kl','forward_kl = None').replace('device="cuda"','device="cpu"')
        ns={};exec(text,ns);ref=ns['TAID'](t_start=.2);ours=Controller()
        for i,loss in enumerate([4.,3.8,4.1,3.5,3.1,3.2,2.9]):
            ref.update_t(torch.tensor(loss),i,100);ours.update(loss,i,100)
            self.assertAlmostEqual(ours.t,float(ref.t),places=6)
            self.assertAlmostEqual(ours.momentum,float(ref.momentum),places=6)
        restored=Controller(**ours.state_dict());restored.update(2.8,7,100);ours.update(2.8,7,100)
        self.assertEqual(restored.state_dict(),ours.state_dict())

    def test_masks_and_truncation(self):
        rows=[dict(source_id=1,input_ids=[1,7,9,2],prompt_len=2),dict(source_id=2,input_ids=[1,8,9,10,11,2],prompt_len=1)]
        rows,cut=normalize(rows,3,3,2);self.assertEqual(cut,1)
        self.assertEqual(rows[1]['input_ids'],[1,8,9,10])
        ids,attn,mask=collate(rows,2)
        self.assertEqual(int(mask.sum()),6);self.assertTrue(mask[0,0]) # prompt included
        self.assertEqual(int(attn[0,3]),1) # true EOS must not be padding
        _,_,mask=collate(rows,2,stress=True,prompt_cap=3,response_cap=3)
        self.assertEqual(int(mask.sum()),10)

    def test_schedule(self):
        per,steps=schedule(10949,384,5)
        self.assertEqual(per,28);self.assertEqual(steps,list(range(14,141,14)))

    def test_optimizer_resume(self):
        m=torch.nn.Linear(3,5);o=torch.optim.AdamW(parameter_groups(m),lr=1e-4)
        s=torch.optim.lr_scheduler.CosineAnnealingLR(o,10);c=Controller()
        def step(m,o,s,c,i):
            o.zero_grad();x=m(torch.randn(2,3));target=torch.randn_like(x)
            v=loss_sum(x,target,torch.ones(2),c.t)/2
            v.backward();o.step();s.step();c.update(float(v.detach()),i,10)
        step(m,o,s,c,0)
        weights=copy.deepcopy(m.state_dict());optim=copy.deepcopy(o.state_dict());sched=copy.deepcopy(s.state_dict());state=c.state_dict();rng=torch.get_rng_state()
        step(m,o,s,c,1)
        n=torch.nn.Linear(3,5);n.load_state_dict(weights);oo=torch.optim.AdamW(parameter_groups(n),lr=1e-4);ss=torch.optim.lr_scheduler.CosineAnnealingLR(oo,10)
        oo.load_state_dict(optim);ss.load_state_dict(sched);cc=Controller(**state);torch.set_rng_state(rng)
        step(n,oo,ss,cc,1)
        for a,b in zip(m.parameters(),n.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)
        self.assertEqual(c.state_dict(),cc.state_dict())


if __name__=='__main__':unittest.main(verbosity=2)
