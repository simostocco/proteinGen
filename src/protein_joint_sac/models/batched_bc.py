"""Literal ragged-prefix batching of reference actors; unchanged model state-dict and geometry density."""
import math
from contextlib import nullcontext
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from ..chemistry.reference_adapter import chemistry_table,load

def rope(x,positions):
    # Trigonometry FP32 even under BF16; positions include each example's true query index.
    inv=10000.**(-torch.arange(0,x.shape[-1],2,device=x.device,dtype=torch.float32)/x.shape[-1])
    a=positions.float()[...,None]*inv;c=a.cos()[:,None];s=a.sin()[:,None]
    y=x.float();return torch.stack([y[...,::2]*c-y[...,1::2]*s,y[...,::2]*s+y[...,1::2]*c],-1).flatten(-2).to(x.dtype)

def collate(states,device):
    B=len(states);L=max(s.t for s in states);x=torch.zeros(B,L,3,dtype=torch.float64);valid=torch.zeros(B,L,dtype=torch.bool);ids=torch.zeros(B,L,dtype=torch.long)
    for i,s in enumerate(states):
        if s.ca.device.type!='cpu':raise ValueError('collate CPU literal prefixes only')
        x[i,:s.t]=s.ca;valid[i,:s.t]=s.valid
        ids[i,:s.t]=torch.tensor([load().INDEX[a] for a in s.residues])
    return {k:v.to(device,non_blocking=True) for k,v in {'x':x,'valid':valid,'ids':ids,'t':torch.tensor([s.t for s in states]),'N':torch.tensor([s.N for s in states])}.items()}

def features_batch(batch,chemistry=True,sequence_only=False):
    x=batch['x'];valid=batch['valid'];ids=batch['ids'];ts=batch['t'];Ns=batch['N'];B,L,_=x.shape;dev=x.device;present=torch.arange(L,device=dev)[None]<ts[:,None]
    valid=valid&present;x=torch.where(valid[...,None],x,0.)
    chem=torch.as_tensor(chemistry_table(),device=dev,dtype=torch.float64)[ids]*present[...,None]
    z=lambda *shape:torch.zeros(*shape,device=dev,dtype=torch.float64)
    bonds=z(B,L);bends=z(B,L);tors=z(B,L);frame=z(B,L,3,3);bm=torch.zeros(B,L,device=dev,dtype=torch.bool);am=bm.clone();tm=bm.clone();fv=bm.clone()
    if not sequence_only:
        if L>=2:
            b=x[:,1:]-x[:,:-1];bn=torch.linalg.vector_norm(b,dim=-1);bm[:,1:]=valid[:,1:]&valid[:,:-1];bonds[:,1:]=torch.where(bm[:,1:],bn,0.)
        if L>=3:
            a=b[:,:-1];c=b[:,1:];an=bn[:,:-1];cn=bn[:,1:];cross=torch.linalg.cross(a,c);crossn=torch.linalg.vector_norm(cross,dim=-1)
            av=valid[:,:-2]&valid[:,1:-1]&valid[:,2:]&(an>1e-8)&(cn>1e-8);am[:,2:]=av
            bends[:,2:]=torch.where(av,torch.atan2(torch.where(av,crossn,0.),torch.where(av,-(a*c).sum(-1),1.)),0.)
            vv=av&(crossn>1e-8);fv[:,2:]=vv;ex=c/cn.clamp_min(1e-10)[...,None];ez=cross/crossn.clamp_min(1e-10)[...,None];ey=torch.linalg.cross(ez,ex)
            frame[:,2:]=torch.where(vv[...,None,None],torch.stack([ex,ey,ez],dim=2),0.)
        if L>=4:
            u=b[:,1:-1]/bn[:,1:-1].clamp_min(1e-10)[...,None];v=-b[:,:-2];v=v-(v*u).sum(-1)[...,None]*u;w=b[:,2:]-(b[:,2:]*u).sum(-1)[...,None]*u
            vn=torch.linalg.vector_norm(v,dim=-1);wn=torch.linalg.vector_norm(w,dim=-1)
            tv=valid[:,:-3]&valid[:,1:-2]&valid[:,2:-1]&valid[:,3:]&(bn[:,:-2]>=1e-10)&(bn[:,1:-1]>=1e-10)&(bn[:,2:]>=1e-10)&(vn>=1e-10)&(wn>=1e-10);tm[:,3:]=tv
            vh=v/vn.clamp_min(1e-10)[...,None];wh=w/wn.clamp_min(1e-10)[...,None];si=(torch.linalg.cross(u,vh)*wh).sum(-1);co=(vh*wh).sum(-1)
            tors[:,3:]=torch.where(tv,torch.atan2(torch.where(tv,si,0.),torch.where(tv,co,1.)),0.)
    # Prefix geometry is evaluated in FP64; neural pair features cast after invariants.
    if sequence_only:
        crowd=z(B,L);exposure=z(B,L);packing=z(B,L);pair=None
    else:
        vec=x[:,None,:,:]-x[:,:,None,:];dist=torch.linalg.vector_norm(vec,dim=-1);pv=valid[:,:,None]&valid[:,None,:];direction=vec/dist.clamp_min(1e-8)[...,None]
        ori=torch.einsum('biac,bijc->bija',frame,direction);rot=torch.einsum('biac,bjdc->bijad',frame,frame).flatten(-2)
        idx=torch.arange(L,device=dev);long=(idx[:,None]-idx[None,:]).abs()>=3;w=torch.relu(1-dist/8.).square()*pv*long
        crowd=w.sum(-1);exposure=torch.where(valid,1/(1+crowd),0.);ci=load().CHEM_NAMES.index('carbon_only_sidechain');packing=(w*chem[:,:,ci,None]*chem[:,None,:,ci]).sum(-1)
        rbf=torch.exp(-((dist.float()[...,None]-torch.linspace(0,20,8,device=dev))/3.).square());sep=((idx[:,None]-idx[None,:])/500.).expand(B,L,L)[...,None]
        chemical=(chem.float()@chem.float().transpose(-1,-2)/100.)[...,None] if chemistry else torch.zeros(B,L,L,1,device=dev)
        masks=torch.stack([pv,fv[:,:,None].expand(B,L,L),fv[:,None,:].expand(B,L,L)],-1).float()
        pair=(torch.cat([rbf,sep,ori.float(),rot.float(),chemical,masks],-1),pv)
    term=z(B,L,3)
    if L:
        term[:,0,0]=(ts>0);ix=(ts-1).clamp_min(0);rows=torch.arange(B,device=dev);term[rows,ix,1]=((ts==Ns)&(ts>0)).double();term[rows,ix,2]=((ts<Ns)&(ts>0)).double()
    ch=chem if chemistry else torch.zeros_like(chem)
    if sequence_only:feat=torch.cat([ch,z(B,L,12),term,z(B,L,1)],-1)
    else:
        geom=torch.stack([bonds,bends,tors.sin()*tm,tors.cos()*tm,crowd],-1);masks=torch.stack([bm,am,tm],-1)
        feat=torch.cat([ch,geom,masks,fv[...,None],torch.stack([crowd,exposure,packing if chemistry else torch.zeros_like(packing)],-1),term,valid[...,None]],-1)
    return feat.float(),pair,present

def block_forward(block,x,bias,positions):
    B,L,D=x.shape;H=block.c.heads;dh=D//H
    qkv=block.qkv(block.norm(x)).view(B,L,3,H,dh);q,k,v=[qkv[:,:,i].transpose(1,2) for i in range(3)]
    q=rope(q,positions);k=rope(k,positions)
    out=F.scaled_dot_product_attention(q,k,v,attn_mask=bias.to(q.dtype),dropout_p=0.,is_causal=False)
    h=x+block.out(out.transpose(1,2).reshape(B,L,D));return h+block.ff(block.norm2(h))

def encode_batch(model,batch,activation_checkpoint=False,prepared=None):
    enc=model.encoder;seqonly=enc.sequence_only;feat,pair,present=features_batch(batch,enc.config.chemistry,seqonly) if prepared is None else prepared
    B,L=present.shape;dtype=enc.tip.dtype;context=enc.context(torch.stack([batch['N']/500.,batch['t']/batch['N']],-1).to(dtype))
    tokens=enc.aa(batch['ids'])+enc.token(feat.to(dtype))+context[:,None,:]
    x=torch.cat([tokens,(enc.tip+context)[:,None]],1);H=enc.config.heads
    bias=torch.zeros(B,H,L+1,L+1,device=x.device,dtype=x.dtype)
    if L and pair is not None and enc.config.pair_bias:
        value,pv=pair;bias[:,:,:L,:L]=enc.pair.net(value).permute(0,3,1,2)*pv[:,None]
    key_valid=torch.cat([present,torch.ones(B,1,device=x.device,dtype=torch.bool)],1);bias=bias.masked_fill(~key_valid[:,None,None,:],float('-inf'))
    positions=torch.cat([torch.arange(L,device=x.device).expand(B,L),batch['t'][:,None]],1)
    for block in enc.blocks:
        if activation_checkpoint and torch.is_grad_enabled():x=checkpoint(block_forward,block,x,bias,positions,use_reentrant=False)
        else:x=block_forward(block,x,bias,positions)
    return enc.final(x[:,-1])

def losses(model,batch,labels,geometry=None,activation_checkpoint=False,prepared=None):
    h=encode_batch(model,batch,activation_checkpoint,prepared);aa=F.cross_entropy(model.aa_head(h).float(),labels,reduction='none')
    if geometry is None:return aa,torch.zeros_like(aa),torch.zeros(len(aa),3,device=aa.device)
    chem=model.chemistry[labels] if model.encoder.config.chemistry else torch.zeros_like(model.chemistry[labels]);p=model.geom(torch.cat([h,model.selected_aa(labels),chem],-1)).float()
    mu=p[:,:3];logstd=p[:,3:].clamp(-5,2);mask=batch['t'][:,None]>=torch.arange(1,4,device=h.device)[None]
    g=geometry.double();d=torch.where(mask[:,0],g[:,0],1.);angle=torch.where(mask[:,1],g[:,1]/math.pi,.5);tau=torch.where(mask[:,2],g[:,2]/math.pi,0.)
    u=torch.stack([d.log(),torch.logit(angle),torch.atanh(tau)],-1)
    jac=torch.stack([u[:,0],math.log(math.pi)-F.softplus(-u[:,1])-F.softplus(u[:,1]),math.log(math.pi)+2*(math.log(2)-u[:,2]-F.softplus(-2*u[:,2]))],-1)
    nll=.5*((u-mu)/logstd.exp()).square()+logstd+.5*math.log(2*math.pi)+jac
    nll=torch.where(mask,nll,0.);geom=nll.sum(-1)/mask.sum(-1).clamp_min(1)
    if not torch.isfinite(aa).all() or not torch.isfinite(geom).all():raise FloatingPointError('nonfinite batched loss/support')
    return aa,geom,nll
