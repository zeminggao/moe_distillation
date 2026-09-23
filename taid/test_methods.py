import copy,contextlib,unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
from expert_methods import ka_indices
from sar_update import update_router
from taid import Controller,loss_sum

class Teacher(torch.nn.Module):
    def __init__(self,policy):
        super().__init__();self.router=torch.nn.Linear(3,8,bias=False)
        self.register_buffer('experts',torch.randn(8,5));self.policy=policy
        self.model=SimpleNamespace(layers=[0])
        self.lm_head=torch.nn.Identity()
        self.model=BackboneView(self)
    def forward(self,input_ids,attention_mask,use_cache=False):
        logits=self.router(input_ids)+.1*torch.randn(*input_ids.shape[:-1],8)
        probabilities=logits.softmax(-1)
        importance=(probabilities*attention_mask[...,None]).sum((0,1))
        if self.policy.collect:self.policy.sums[0].add_(importance.detach())
        balance=0 if self.policy.balance_coefficients is None else (importance*self.policy.balance_coefficients[0]).sum()
        return SimpleNamespace(logits=probabilities@self.experts,balance_loss=balance)
class BackboneView:
    def __init__(self,owner):self.owner=owner;self.layers=[0]
    def __call__(self,**kwargs):
        out=self.owner(**kwargs)
        return SimpleNamespace(last_hidden_state=out.logits,balance_loss=out.balance_loss)

class Student(torch.nn.Module):
    def __init__(self):super().__init__();self.net=torch.nn.Linear(3,5)
    def forward(self,input_ids,**kwargs):return SimpleNamespace(logits=self.net(input_ids))
    @property
    def lm_head(self):return self.net
    def model(self,input_ids,**kwargs):return SimpleNamespace(last_hidden_state=input_ids)
class Policy:
    def __init__(self):
        self.collect=False;self.balance_coefficients=None;self.teacher=Teacher(self)
        self.router_parameters=list(self.teacher.router.parameters())
    def set_batch(self,attention):pass

class Tests(unittest.TestCase):
    def test_ka_sampling(self):
        torch.manual_seed(4);logits=torch.randn(64,8)
        a=ka_indices(logits,1);b=ka_indices(logits,1)
        self.assertTrue(all(len(set(row.tolist()))==7 for row in a))
        self.assertFalse(torch.equal(a,b))
        self.assertTrue(torch.equal(ka_indices(logits,0),logits.topk(7,-1).indices))
    def test_sar_microbatch_and_student_freeze(self):
        torch.manual_seed(7);p=Policy();q=Policy();q.teacher.load_state_dict(p.teacher.state_dict())
        student=Student();before=copy.deepcopy(student.state_dict())
        ids=torch.randn(4,4,3);attn=torch.ones(4,4);mask=torch.tensor([[1,1,1],[1,1,0],[1,0,0],[1,1,1]],dtype=torch.bool)
        # Different-sized randn calls need not match: deterministic zero noise
        # isolates the exact global-token and global-balance gradient comparison.
        with patch('torch.autocast',lambda *a,**k:contextlib.nullcontext()),patch('torch.cuda.synchronize',lambda:None),patch('torch.cuda.get_rng_state',torch.get_rng_state),patch('torch.cuda.set_rng_state',torch.set_rng_state),patch('torch.randn',lambda *shape,**kw:torch.zeros(*shape,**kw)):
            for policy,micro in [(p,1),(q,4)]:
                optimizer=torch.optim.SGD(policy.router_parameters,lr=.01)
                stats=update_router(policy,student,optimizer,ids,attn,mask,int(mask.sum()),micro)
                self.assertGreater(stats['router_grad_norm'],0)
        for a,b in zip(p.router_parameters,q.router_parameters):torch.testing.assert_close(a,b,atol=1e-7,rtol=1e-5)
        for name,value in student.state_dict().items():torch.testing.assert_close(value,before[name]);self.assertTrue(all(x.grad is None for x in student.parameters()))
    def test_router_optimizer_scheduler_resume(self):
        torch.manual_seed(11);router=torch.nn.Linear(3,8);opt=torch.optim.AdamW(router.parameters(),lr=2e-5)
        sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,140)
        x=torch.randn(4,3)
        def step(model,optimizer,scheduler):
            optimizer.zero_grad();loss=(model(x)+torch.randn(4,8)).square().mean();loss.backward();optimizer.step();scheduler.step()
        step(router,opt,sched)
        state=copy.deepcopy(dict(router=router.state_dict(),optimizer=opt.state_dict(),scheduler=sched.state_dict(),rng=torch.get_rng_state()))
        step(router,opt,sched)
        restored=torch.nn.Linear(3,8);restored.load_state_dict(state['router'])
        opt2=torch.optim.AdamW(restored.parameters(),lr=2e-5);opt2.load_state_dict(state['optimizer'])
        sched2=torch.optim.lr_scheduler.CosineAnnealingLR(opt2,140);sched2.load_state_dict(state['scheduler'])
        torch.set_rng_state(state['rng']);step(restored,opt2,sched2)
        for a,b in zip(router.parameters(),restored.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)
        self.assertEqual(sched.state_dict(),sched2.state_dict())
    def test_two_ka_updates_and_restore(self):
        torch.manual_seed(9);net=torch.nn.Linear(3,5);optimizer=torch.optim.AdamW(net.parameters(),lr=1e-4)
        ctrl=Controller();x=torch.randn(2,3);teacher=torch.randn(2,5);mask=torch.ones(2,dtype=torch.bool)
        initial=copy.deepcopy(net.state_dict())
        for index in range(2):
            optimizer.zero_grad();loss=loss_sum(net(x),teacher,mask,ctrl.t)/2;loss.backward();optimizer.step();ctrl.update(float(loss.detach()),index,280)
        self.assertEqual(int(next(iter(optimizer.state.values()))['step']),2)
        self.assertFalse(torch.equal(initial['weight'],net.weight))
        restored=Controller(**ctrl.state_dict());self.assertEqual(restored.state_dict(),ctrl.state_dict())
        self.assertAlmostEqual(ctrl.t,.2+.8/280)
if __name__=='__main__':unittest.main(verbosity=2)
