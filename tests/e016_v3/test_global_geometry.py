import math,copy
import numpy as np
import pytest
import torch
from protein_joint_sac.rl.global_geometry import *
from protein_joint_sac.models.joint_actor import JointActor
from protein_joint_sac.models.dense_geometry_attention import ModelConfig
from protein_joint_sac.geometry.ca_builder import inverse
from protein_joint_sac.data.engineering_fixtures import records
from protein_joint_sac.data.scaled_supervised import canonical
from protein_joint_sac.models.batched_bc import losses

def fixture():
    r=records()[0];x=torch.tensor(canonical(np.array(r["coordinates"])),dtype=torch.float64)
    return x,torch.arange(len(x))%20
def net():
    torch.manual_seed(17);m=JointActor(ModelConfig(width=16,heads=2,layers=1,chemistry=False))
    with torch.no_grad():
        m.geom[-1].bias.copy_(torch.tensor([math.log(3.8),0.,0.,-3.,-1.,-1.]))
        m.geom[-1].weight.mul_(.01)
    return m

def test_native_roundtrip_and_zero_Rg_loss():
    x,ids=fixture();out=x[:3]
    for t in range(3,len(x)):
        g,_=inverse(x[:t+1]);out=torch.cat([out,next_ca(out,g)[None]],0)
    torch.testing.assert_close(x,out,atol=1e-10,rtol=1e-10)
    a,b=global_losses(out,x,3);assert a.item()<1e-20

def test_extended_and_collapsed_geometry_have_complementary_penalties():
    x,ids=fixture();ext=x*4.;collapsed=x*.001
    lr,lc=global_losses(ext,x,3);assert lr.item()>.1
    a,b=global_losses(collapsed,x,3);assert b.item()>1.
    z=collapsed.clone().requires_grad_();global_losses(z,x,3)[1].backward()
    assert torch.isfinite(z.grad).all() and z.grad.norm()>0

def test_global_rigid_transform_invariance():
    x,ids=fixture();g=x.clone();g[8:]+=torch.tensor([2.,3.,-1.])
    q,_=torch.linalg.qr(torch.randn(3,3,dtype=torch.float64));q[:,0]*=torch.linalg.det(q);v=torch.tensor([4.,-8.,7.])
    for a,b in zip(global_losses(g,x,8),global_losses(g@q+v,x@q+v,8)):torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-12)

def test_actual_rollout_has_geometry_and_transformer_gradients():
    x,ids=fixture();m=net();torch.manual_seed(4)
    r=rollout(m,x[:5],ids[:13],len(x),8,x[:13],precision="FP32",recompute=False)
    r["Rg_loss"].backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    assert m.geom[-1].weight.grad.norm()>0
    assert m.encoder.blocks[0].qkv.weight.grad.norm()>0

def test_recompute_matches_eager_outputs_and_gradients():
    x,ids=fixture();m=net();torch.manual_seed(8)
    a=rollout(m,x[:5],ids[:9],len(x),4,x[:9],precision="FP32",recompute=False)
    (a["Rg_loss"]+a["clash_loss"]).backward();grads=[p.grad.clone() if p.grad is not None else None for p in m.parameters()]
    m.zero_grad(set_to_none=True);torch.manual_seed(8)
    b=rollout(m,x[:5],ids[:9],len(x),4,x[:9],precision="FP32",recompute=True)
    (b["Rg_loss"]+b["clash_loss"]).backward()
    torch.testing.assert_close(a["coordinates"],b["coordinates"],atol=0,rtol=0)
    for p,g in zip(m.parameters(),grads):
        if g is not None:torch.testing.assert_close(p.grad,g,atol=1e-7,rtol=1e-5)

def test_native_future_coordinates_are_targets_only():
    x,ids=fixture();m=net();native=x[:11].clone();native[5:]+=100.
    torch.manual_seed(4);a=rollout(m,x[:5],ids[:11],len(x),6,x[:11],precision="FP32")
    torch.manual_seed(4);b=rollout(m,x[:5],ids[:11],len(x),6,native,precision="FP32")
    torch.testing.assert_close(a["coordinates"],b["coordinates"],atol=0,rtol=0)
    logits,_=conditional_parameters(m,x[:5],ids[:5],len(x),ids[5],precision="FP32")
    other,_=conditional_parameters(m,x[:5],ids[:5],len(x),(ids[5]+1)%20,precision="FP32")
    torch.testing.assert_close(logits,other,atol=0,rtol=0)

def test_coordinate_derivative_matches_finite_difference():
    x,ids=fixture();g=inverse(x[:6])[0].requires_grad_()
    def objective(v):
        out=torch.cat([x[:5],next_ca(x[:5],v)[None]],0)
        return global_losses(out,x[:6]*1.2,5)[0]
    objective(g).backward()
    for j in range(3):
        eps=1e-5;plus=g.detach().clone();minus=plus.clone();plus[j]+=eps;minus[j]-=eps
        fd=(objective(plus)-objective(minus))/(2*eps)
        torch.testing.assert_close(g.grad[j],fd,atol=1e-8,rtol=1e-4)

def test_zero_auxiliary_weights_preserve_base_loss_gradients():
    x,ids=fixture();m=net();b=prefix_batch(x[:8],ids[:8],len(x));y=ids[8:9];g=inverse(x[:9])[0][None]
    aa,geo,_=losses(m,b,y,g,geometry_fp32=True);base=(aa+geo).mean();base.backward()
    grads=[p.grad.clone() if p.grad is not None else None for p in m.parameters()]
    m.zero_grad(set_to_none=True);aa,geo,_=losses(m,b,y,g,geometry_fp32=True)
    loss=(aa+geo).mean()+0.*sum(global_losses(x[:9],x[:9],8));loss.backward()
    for p,v in zip(m.parameters(),grads):
        if v is not None:torch.testing.assert_close(p.grad,v,atol=0,rtol=0)
