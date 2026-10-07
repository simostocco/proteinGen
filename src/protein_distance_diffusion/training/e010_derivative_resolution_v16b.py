"""Fixed-state derivative resolution; no minimizer or solver invocation."""
import numpy as np
import torch
from . import e010_sequential_teacher_v16a as historical

GRID=(1e-10,3e-10,1e-9,3e-9,1e-8,3e-8,1e-7,3e-7,1e-6,3e-6,1e-5,3e-5,1e-4)
PHASES=(0.,.7,1.3)
EPSILONS=(1e-6,1e-5,1e-4)
ATOL,RTOL=1e-8,1e-5


def state(n):
    return np.sin(np.arange(3*n)*.7)*.03


def direction(n,phase):
    d=np.cos(np.arange(3*n)+phase)
    return d/np.linalg.norm(d)


def centered(fun,x,d,h):
    plus=np.atleast_1d(fun(x+h*d)); minus=np.atleast_1d(fun(x-h*d))
    return (plus-minus)/(2*h),plus,minus


def errors(fd,exact):
    absolute=np.abs(fd-exact)
    relative=absolute/np.maximum(np.abs(exact),1e-300)
    digits=np.clip(-np.log10(np.maximum(relative,1e-16)),0,16)
    return dict(absolute=absolute,relative=relative,digits=digits,sign_agreement=np.sign(fd)==np.sign(exact),
                passes=absolute<=ATOL+RTOL*np.abs(exact))


def physical(step,x,d,h):
    eligible=step.eligible.numpy()
    intended=(.1*h*d.reshape(step.shape)*eligible[...,None])
    reference=step.point(x)[2].detach().numpy().copy()
    actual=step.point(x+h*d)[2].detach().numpy()-reference
    norms=np.linalg.norm(intended,axis=-1)[step.context.b['mask'].numpy()]
    actual_norms=np.linalg.norm(actual,axis=-1)[step.context.b['mask'].numpy()]
    return dict(intended_max_angstrom=float(norms.max()),intended_rms_angstrom=float(np.sqrt(np.mean(norms**2))),
                stored_max_angstrom=float(actual_norms.max()),stored_rms_angstrom=float(np.sqrt(np.mean(actual_norms**2))),
                unchanged_coordinate_fraction=float((actual==0).mean()))


def direct_analytic(step,x,d):
    """Forward directional autograd independently through complete geometry.

    Historical path constructs a coordinate Jacobian and applies a masked linear
    chain. This path uses scalar-vector directional JVP through the full function.
    Also check reverse scalar contractions for each whole constraint family.
    """
    tensor=torch.from_numpy(x.copy()); tangent=torch.from_numpy(d.copy())
    def fn(z):
        p=step.current+.1*z.reshape(step.shape)*step.eligible[...,None]
        return torch.cat(((step.context.values(p)['local']/step.context.baseline['local']).reshape(1),
                          -step.context.constraints(p)))
    _,jvp=torch.autograd.functional.jvp(fn,tensor,tangent,strict=True)
    return jvp.detach().numpy()


def proposed_h(step,x,d):
    # Fixed normalized-variable rule, not adapted per direction/outcome.
    return 1e-4


def audit_direction(step,x,d,phase):
    exact=np.r_[step.jac(x)@d, step.cjac(x)@d]
    independently=direct_analytic(step,x,d)
    disagreement=np.abs(exact-independently)
    families={'objective':(0,1),'aligned':(1,2),'continuous_chiral':(2,3),
              **{k:(lo+3,hi+3) for k,(lo,hi) in step.context.quartets.slices.items()}}
    analytic_pass=bool(np.all(disagreement<=1e-12+1e-10*np.abs(exact)))
    scalar_checks={}
    for name,(lo,hi) in families.items():
        z=torch.from_numpy(x.copy()).requires_grad_()
        p=step.current+.1*z.reshape(step.shape)*step.eligible[...,None]
        out=torch.cat(((step.context.values(p)['local']/step.context.baseline['local']).reshape(1),
                       -step.context.constraints(p)))
        weights=torch.cos(torch.arange(hi-lo,dtype=torch.float64)*.7)
        weights=weights/weights.norm()
        gf=torch.autograd.grad((out[lo:hi]*weights).sum(),z)[0].numpy()
        scalar=float(gf@d); expected=float(exact[lo:hi]@weights.numpy())
        agrees=abs(scalar-expected)<=1e-12+1e-10*abs(expected)
        scalar_checks[name]=dict(direct_scalar_autograd=scalar,full_jacobian_contraction=expected,
                                 absolute_difference=abs(scalar-expected),passes=agrees)
    def fn(q):
        return np.r_[step.fun(q),step.cfun(q)]
    f0=fn(x)
    arrays=dict(x=x.copy(),direction=d.copy(),analytic=exact,independent_analytic=independently,f0=f0)
    samples=[]; historical_checks=[]; richardson=[]
    for h in GRID:
        fd,plus,minus=centered(fn,x,d,h)
        e=errors(fd,exact); difference=np.abs(plus-minus)
        roundoff_scale=np.finfo(np.float64).eps*np.maximum(np.maximum(np.abs(plus),np.abs(minus)),np.abs(f0))
        families_stats={}
        for name,(lo,hi) in families.items():
            tol=ATOL+RTOL*np.abs(exact[lo:hi]); worst=int(np.argmax(e['absolute'][lo:hi]/tol))+lo
            families_stats[name]=dict(rows=hi-lo,failures=int((~e['passes'][lo:hi]).sum()),
              maximum_absolute_error=float(e['absolute'][lo:hi].max()),
              maximum_relative_error=float(e['relative'][lo:hi].max()),
              median_relative_error=float(np.median(e['relative'][lo:hi])),
              minimum_digits=float(e['digits'][lo:hi].min()),median_digits=float(np.median(e['digits'][lo:hi])),
              sign_agreement_fraction=float(e['sign_agreement'][lo:hi].mean()),
              worst_index=worst,analytic=float(exact[worst]),fd=float(fd[worst]),
              f0=float(f0[worst]),f_plus=float(plus[worst]),f_minus=float(minus[worst]),
              numerator=float(difference[worst]),roundoff_local_scale=float(roundoff_scale[worst]),
              numerator_to_roundoff=float(difference[worst]/max(roundoff_scale[worst],1e-300)),
              tolerance=float(ATOL+RTOL*abs(exact[worst])),absolute_error=float(e['absolute'][worst]),
              relative_error=float(e['relative'][worst]),
              maximum_tolerance_ratio=float((e['absolute'][lo:hi]/tol).max()))
        samples.append(dict(h=h,physical=physical(step,x,d,h),passes_all=bool(e['passes'].all()),families=families_stats))
        label=format(h,'.0e')
        arrays.update({f'{label}_plus':plus,f'{label}_minus':minus,f'{label}_fd':fd,
                       f'{label}_absolute':e['absolute'],f'{label}_relative':e['relative'],
                       f'{label}_sign':e['sign_agreement'],f'{label}_digits':e['digits']})
        half,_,_=centered(fn,x,d,h/2)
        extrapolated=(4*half-fd)/3
        richardson.append(dict(h=h,half_h=h/2,maximum_Dh_Dhalf_difference=float(np.abs(fd-half).max()),
          maximum_extrapolated_absolute_error=float(np.abs(extrapolated-exact).max()),
          half_passes_all=bool(errors(half,exact)['passes'].all())))
        arrays[f'{label}_half_fd']=half
        if h in EPSILONS:
            historical_checks.append(dict(phase=phase,historical_h=h,scalar_objective=families_stats['objective'],
                    vector_constraint_families={k:v for k,v in families_stats.items() if k!='objective'},
                    vector_failures=sum(v['failures'] for k,v in families_stats.items() if k!='objective'),
                    passes_all=bool(e['passes'].all()),physical=samples[-1]['physical']))
    return dict(phase=phase,state_sha256=historical.digest(x),direction_sha256=historical.digest(d),
                analytic_paths_pass=analytic_pass,
                float32_chain_scale=float((.1*step.eligible).max()),
                chain_bias_factor=float(np.float32(.1))/.1,
                chain_factor_model_max_error=float(np.abs(exact[3:]-independently[3:]*(float(np.float32(.1))/.1)).max()),
                analytic_max_disagreement=float(disagreement.max()),scalar_analytic_checks=scalar_checks,families=families,samples=samples,
                historical_checks=historical_checks,richardson=richardson,
                stable_grid_values=[s['h'] for s in samples if s['passes_all']],
                proposed_future_h=proposed_h(step,x,d)),arrays
