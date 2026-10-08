"""Controlled-context isolation, equivariance and baseline-parity preflight."""
import inspect
import json
import numpy as np
import torch
import train_e010_student_direction_observability_v1 as t
e=t.e


def main():
    e.initialize();t.prepare();row=e.rows()[0];item=t.identity_data(row)
    p,m,d,_,c=t.make_batch([(item,0,0)],'A','cpu')
    torch.manual_seed(610301);baseline=t.old.Student().eval()
    torch.manual_seed(610301);reproduction=t.ContextStudent('A').eval()
    for name,value in baseline.state_dict().items():assert torch.equal(value,reproduction.state_dict()[name])
    assert torch.equal(baseline(p,m),reproduction(p,m,c))
    o1=torch.optim.AdamW(baseline.parameters(),lr=3e-4,weight_decay=1e-4)
    o2=torch.optim.AdamW(reproduction.parameters(),lr=3e-4,weight_decay=1e-4)
    loss,_,_=t.old.action_loss(baseline,p,m,d);loss.backward();torch.nn.utils.clip_grad_norm_(baseline.parameters(),1.);o1.step()
    loss,_=t.loss(reproduction,p,m,d,c);loss.backward();torch.nn.utils.clip_grad_norm_(reproduction.parameters(),1.);o2.step()
    for name,value in baseline.state_dict().items():assert torch.equal(value,reproduction.state_dict()[name])
    q,_=torch.linalg.qr(torch.tensor([[.2,.3,.8],[.6,.9,.1],[.4,.1,.7]],dtype=torch.float64))
    if torch.det(q)<0:q[:,0]*=-1
    for variant in t.VARIANTS:
        torch.manual_seed(610301);model=t.ContextStudent(variant).eval();assert sum(v.numel() for v in model.parameters())==118723
        torch.nn.init.normal_(model.local.head.weight,0,.001)
        available=torch.from_numpy(t.context(item,0,variant)[None]);first=model(p,m,available)
        rotated=model(p@q+torch.tensor([5.,2.,-3.]),m,available)
        assert torch.allclose(rotated,first@q,atol=1e-7,rtol=1e-5)
        guarded,_=t.old.pilot.project(first.detach().numpy());valid=t.old.local_representation(p,m)['eligible'].numpy()
        assert np.linalg.norm(guarded,axis=-1).max()<=.1 and np.all(guarded[~valid]==0)
        changed=dict(item);changed['conditions']=[dict(v,target=v['target']+1000.) for v in item['conditions']]
        assert np.array_equal(t.context(item,0,variant),t.context(changed,0,variant))
    with np.load(e.ROOT/'representations/identity_0000.npz',allow_pickle=False) as z:
        source=torch.from_numpy(item['conditions'][0]['source'][None])
        f=t.old.local_representation(source,m)['features'][0].float().numpy()
        assert np.array_equal(f,z['50_coarse_output_context'][0,:,6:])
        rotated=t.old.local_representation(source@q+torch.tensor([5.,2.,-3.]),m)['features'][0].float().numpy()
        assert np.allclose(f,rotated,atol=1e-6,rtol=1e-5)
    signature=list(inspect.signature(t.ContextStudent.forward).parameters);assert signature==['self','p','mask','context']
    e.write(t.ROOT/'input_tests.json',dict(passed=True,trainable_parameters_every_variant=118723,
        baseline_initial_weights_forward_and_CPU_optimizer_step_bitwise=True,SO3_translation_equivariance=True,
        cap_and_eligibility=True,context_unchanged_under_target_substitution=True,
        E_source_features_equal_observability_archive=True,forward_signature=signature))
    print(json.dumps(dict(passed=True,variants=list(t.VARIANTS))),flush=True)


if __name__=='__main__':main()
