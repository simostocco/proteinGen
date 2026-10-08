"""Validation-gated, fixed-capacity observable-context controls; D: only."""
import argparse
import concurrent.futures
import json
import multiprocessing
import time
import numpy as np
import torch
import e010_student_direction_observability_v1 as e
e.storage()
old=e.old
ROOT=e.ROOT/'phase_C'
VARIANTS={'A':0,'B':3,'C':21,'D':24,'E':41}
DATA=None
MODEL=None
MODEL_SHA=None


class ContextStudent(old.Student):
    def __init__(self,variant):
        super().__init__();self.variant=variant;dimension=VARIANTS[variant]
        rng=np.random.default_rng(610301+777)
        projection=np.linalg.qr(rng.normal(size=(64,dimension)))[0].T if dimension else np.zeros((0,64))
        projection=projection*np.sqrt(64/max(dimension,1))*.25
        self.register_buffer('context_projection',torch.from_numpy(projection.astype(np.float32)))

    def forward(self,p,mask,context=None):
        out=old.local_representation(p.detach(),mask);valid=out['eligible']
        h=self.local.input(out['features'].float())*valid[...,None]
        if self.variant!='A':h=(h+context.float()@self.context_projection)*valid[...,None]
        for block in self.local.blocks:h=block(h,valid)
        u=self.local.head(h).double()*valid[...,None]
        d=torch.einsum('bnij,bnj->bni',out['frame'],u);norm=d.norm(dim=-1,keepdim=True).clamp_min(1e-30)
        return d*torch.clamp(.1/norm,max=1.)*valid[...,None]


def prepare():
    assert json.loads((e.ROOT/'results_AB.json').read_text())['phase_C_gate']['winning_context']=='denoiser_geometry'
    for name in ('','models','evaluation','logs'):(ROOT/name).mkdir(parents=True,exist_ok=True)
    protocol=dict(variants={'A':'geometry baseline','B':'geometry + known condition','C':'geometry + primary-sequence residue identity',
        'D':'geometry + sequence + condition','E':'geometry + archived denoiser-output local geometry only'},
        same_capacity=118723,width=64,blocks=2,seed=610301,context_projection='fixed orthonormal rank-preserving map to 64 channels, seed611078, RMS scale0.25; injected before original blocks',
        categorical_encoding='one-hot known condition or 20 canonical amino acids plus unknown; each context has unit input RMS; D gives condition and sequence equal energy',
        E_encoding='41 frozen local geometry channels computed from source denoiser prediction and mask only; TRAIN reference-panel mean/std, floor1e-3',
        training_protocol='exact fc76211 identity-first sampler, seed, optimizer, batch8, 3000-6000 updates, check1000, patience2, retention delta0.002',
        checkpoint_monitor='original 60 validation identity offline shielded K16 teacher-repair retention; X permitted for offline checkpoint scoring only',
        primary_evaluation='raw unshielded one-step and recursive K16, no X-based deployment shield',
        no_target_input=True,no_new_teacher_actions=True,no_capacity_increase=True,no_SLSQP=True)
    if (ROOT/'protocol.json').exists():assert json.loads((ROOT/'protocol.json').read_text())==protocol
    else:e.write(ROOT/'protocol.json',protocol)


def identity_data(row):
    item=old.load_identity(row)
    with np.load(e.ROOT/'representations'/f'identity_{row["index"]:04d}.npz',allow_pickle=False) as z:item['sequence_tokens']=z['sequence_tokens'].copy()
    transform=json.loads((e.ROOT/'panel/transforms.json').read_text())['coarse_output_context']
    mean=np.asarray(transform['mean'][6:],np.float32);scale=np.asarray(transform['scale'][6:],np.float32)
    for c in item['conditions']:
        with torch.no_grad():features=old.local_representation(torch.from_numpy(c['source'][None]),torch.from_numpy(c['mask'][None]))['features'][0].float().numpy()
        c['denoiser_context']=(features-mean)/scale
    return item


def context(item,ci,variant):
    n=len(item['sequence_tokens'])
    if variant=='A':return np.zeros((n,0),np.float32)
    if variant=='E':return item['conditions'][ci]['denoiser_context']
    seq=np.eye(21,dtype=np.float32)[item['sequence_tokens']]
    cond=np.broadcast_to(np.eye(3,dtype=np.float32)[ci],(n,3))
    if variant=='B':return cond*np.sqrt(3)
    if variant=='C':return seq*np.sqrt(21)
    return np.concatenate([seq*np.sqrt(12),cond*np.sqrt(12)],axis=1)


def make_batch(selected,variant,device):
    samples=[old.sample(item,ci,step) for item,ci,step in selected];p,m,d,target=old.batch(samples,device)
    ctx=np.zeros((len(samples),p.shape[1],VARIANTS[variant]),np.float32)
    for i,(item,ci,_) in enumerate(selected):ctx[i,:len(item['sequence_tokens'])]=context(item,ci,variant)
    return p,m,d,target,torch.from_numpy(ctx).to(device)


def loss(model,p,m,label,c):
    d=model(p,m,c);valid=old.local_representation(p,m)['eligible']
    action=(((d-label).square().sum(-1)*valid).sum(1)/valid.sum(1).clamp_min(1)/.1**2).mean()
    energy=((label.square().sum(-1)*valid).sum(1)/valid.sum(1).clamp_min(1)/.1**2).mean()
    return action,energy


@torch.no_grad()
def probe(model,identities,device):
    samples=[(DATA[sid],ci,step) for sid in identities for ci in range(3) for step in (0,7,15)]
    errors=[];energies=[]
    for start in range(0,len(samples),8):
        p,m,d,_,c=make_batch(samples[start:start+8],model.variant,device);error,energy=loss(model,p,m,d,c)
        errors.append(float(error)*len(p));energies.append(float(energy)*len(p))
    return dict(normalized_imitation=sum(errors)/len(samples),zero_predictor_error=sum(energies)/len(samples),error_to_zero_baseline=sum(errors)/sum(energies),samples=len(samples))


def save(path,model,update):
    temporary=path.with_suffix('.tmp')
    with temporary.open('wb') as f:torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},variant=model.variant,update=update,
        splits_sha256=old.sha(old.ROOT/'splits.json'),protocol_sha256=old.sha(ROOT/'protocol.json')),f)
    temporary.replace(path)


def init_evaluator(path):
    global MODEL,MODEL_SHA
    e.initialize();payload=torch.load(path,map_location='cpu',weights_only=False);MODEL=ContextStudent(payload['variant']).eval();MODEL.load_state_dict(payload['model'])
    assert payload['splits_sha256']==old.sha(old.ROOT/'splits.json')
    MODEL_SHA=old.sha(path)


@torch.no_grad()
def predict(p,m,c):
    start=time.perf_counter();d=MODEL(p,m,c).numpy();d,guard=old.pilot.project(d)
    return d,dict(policy_seconds=time.perf_counter()-start,guard=guard)


def evaluate_identity(task):
    row,quick,folder=task;destination=ROOT/'evaluation'/folder/f'identity_{row["index"]:04d}.json'
    if destination.exists():
        saved=json.loads(destination.read_text());assert all(r['model_sha256']==MODEL_SHA for r in saved);return saved
    item=identity_data(row);results=[];arrays={}
    with torch.no_grad():
        for ci,c in enumerate(item['conditions']):
            b={key:torch.from_numpy(c[name][None]) for key,name in [('pg','P0'),('source','source'),('target','target'),('mask','mask')]}
            ctx=old.pilot.Context(b);baseline=old.metrics(ctx,b['pg']);available=torch.from_numpy(context(item,ci,MODEL.variant)[None]);one=[];one_ds=[]
            if not quick:
                for step in range(16):
                    state,_,teacher,_=old.sample(item,ci,step);p=torch.from_numpy(state[None]);raw,policy=predict(p,b['mask'],available)
                    a=old.action_metrics(ctx,p,raw,policy);failed=not a['after']['safety']['passes'] or a['cap_violations']>0 or a['eligibility_violations']>0
                    comparison=old.frozen.old.comparison(raw,teacher[None],old.local_representation(p,b['mask'])['eligible'].numpy())
                    td=torch.from_numpy(teacher[None]);error=torch.from_numpy(raw-teacher[None]);endpoint=p+td
                    constraints=ctx.constraints(endpoint).numpy();slack=-constraints;slack[:2]+=1e-8
                    derivative=((ctx.constraints(endpoint+1e-4*error)-ctx.constraints(endpoint-1e-4*error))/(2e-4)).numpy()
                    risk=float((np.maximum(derivative,0)/np.maximum(slack,1e-12)).max())
                    one.append(dict(step=step+1,inference_seconds=policy['policy_seconds'],before=old.metrics(ctx,p),teacher_after=old.metrics(ctx,endpoint),hard_failed=failed,
                        directional_risk_ratio=risk,**comparison,**a));one_ds.append(raw)
            p=b['pg'].clone();steps=[];ds=[]
            for step in range(16):
                start=time.perf_counter()
                raw,policy=predict(p,b['mask'],available)
                d,cert=old.shield(ctx,p,raw) if quick else (raw,None)
                inference_seconds=time.perf_counter()-start
                a=old.action_metrics(ctx,p,d,policy,cert);a['hard_failed']=not a['after']['safety']['passes'] or a['cap_violations']>0 or a['eligibility_violations']>0
                ds.append(d);p=p+torch.from_numpy(d);steps.append(dict(step=step+1,inference_seconds=inference_seconds,**a))
            results.append(dict(identity=row['sample_id'],model_sha256=MODEL_SHA,condition=c['condition'],length=row['length'],length_bin=row['stratum'],baseline=baseline,
                teacher_final=c['teacher']['final'],one_step=one,rollouts={('shielded' if quick else 'raw'):dict(final=steps[-1]['after'],steps=steps)}))
            if not quick:arrays[f'{c["condition"]}_raw_one_delta']=np.stack(one_ds);arrays[f'{c["condition"]}_raw_rollout_delta']=np.stack(ds)
    if not quick:
        old.frozen.old.atomic_npz(destination.with_suffix('.npz'),**arrays)
        for result in results:result.update(action_archive=str(destination.with_suffix('.npz')),action_archive_sha256=old.sha(destination.with_suffix('.npz')))
    e.write(destination,results);return results


def evaluate(path,identities,destination,quick=False):
    if destination.exists():
        result=json.loads(destination.read_text());assert result['model_sha256']==old.sha(path);return result
    folder=destination.stem;(ROOT/'evaluation'/folder).mkdir(parents=True,exist_ok=True);by_id={r['sample_id']:r for r in e.rows()};start=time.perf_counter();records=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=8,mp_context=multiprocessing.get_context('spawn'),initializer=init_evaluator,initargs=(str(path),)) as pool:
        for i,rows in enumerate(pool.map(evaluate_identity,[(by_id[sid],quick,folder) for sid in identities]),1):
            records.extend(rows)
            if i%50==0:print(json.dumps(dict(evaluation=folder,identities=i,total=len(identities))),flush=True)
    mode='shielded' if quick else 'raw';result=dict(identities=identities,quick=quick,model_sha256=old.sha(path),rows=records,
        summary={mode:old.summary(records,mode)},wall_seconds=time.perf_counter()-start)
    e.write(destination,result);return result


def train(variant):
    global DATA
    e.initialize();prepare();assert torch.cuda.is_available();split=json.loads((old.ROOT/'splits.json').read_text())
    DATA={r['sample_id']:identity_data(r) for r in e.rows()}
    config=old.CONFIG;torch.manual_seed(config['seed']);torch.cuda.manual_seed_all(config['seed']);rng=np.random.default_rng(config['seed'])
    device=torch.device('cuda');model=ContextStudent(variant).to(device);assert sum(p.numel() for p in model.parameters())==118723
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay']);train_ids=split['splits']['train']
    train_probe=sorted(train_ids,key=lambda sid:old.rank(f'trainprobe|{sid}'))[:60];monitor=split['monitor'];history=[];best=-float('inf');stale=0;best_update=None
    start=time.perf_counter();gpu_seconds=0.
    for update in range(1,config['maximum_updates']+1):
        selected=rng.choice(train_ids,size=8,replace=True);samples=[(DATA[sid],int(rng.integers(3)),int(rng.integers(16))) for sid in selected]
        p,m,label,_,c=make_batch(samples,variant,device);optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize();begin=time.perf_counter()
        imitation,_=loss(model,p,m,label,c);imitation.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(norm)
        optimizer.step();torch.cuda.synchronize();gpu_seconds+=time.perf_counter()-begin
        if update%100==0:print(f'TRAIN {variant} {update} imitation={float(imitation):.6f}',flush=True)
        if update%1000==0:
            path=ROOT/'models'/f'{variant}_{update:05d}.pt';save(path,model,update);model.eval()
            train_loss=probe(model,train_probe,device);val_loss=probe(model,monitor,device)
            evaluation=evaluate(path,monitor,ROOT/'evaluation'/f'{variant}_{update:05d}_monitor.json',quick=True);retention=evaluation['summary']['shielded']['RMSE_retention']
            history.append(dict(update=update,validation_retention=retention,train_probe=train_loss,validation_probe=val_loss,model_sha256=old.sha(path)))
            if retention>best+.002:best=retention;best_update=update;stale=0;save(ROOT/'models'/f'{variant}_best.pt',model,update)
            else:stale+=1
            e.write(ROOT/f'{variant}_training_progress.json',dict(history=history,best_update=best_update));print(f'VALIDATION {variant} {update} retention={retention:.6f} stale={stale}',flush=True)
            if update>=3000 and stale>=2:break
            model.train()
    e.write(ROOT/f'{variant}_training.json',dict(history=history,best_update=best_update,updates=update,wall_seconds=time.perf_counter()-start,
        GPU_training_seconds=gpu_seconds,parameters=118723,source_sha256=old.sha(__file__),split_sha256=old.sha(old.ROOT/'splits.json')))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','train','evaluate']);parser.add_argument('--variant',choices=VARIANTS,default='A');parser.add_argument('--split',choices=['validation','test'],default='validation');args=parser.parse_args()
    prepare()
    if args.mode=='train':train(args.variant)
    elif args.mode=='evaluate':
        e.initialize();split=json.loads((old.ROOT/'splits.json').read_text())['splits'][args.split]
        evaluate(ROOT/'models'/f'{args.variant}_best.pt',split,ROOT/'evaluation'/f'{args.variant}_{args.split}.json')


if __name__=='__main__':main()
