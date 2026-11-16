"""CPU contracts before any admission inference or production optimization."""
from paired import *
from analyze import ci,primary_supported
def main():
    verify();checks=[]
    a=read(ARMS['A']/'TRAINING_CONFIGURATION_AND_SEEDS.json');b=read(ARMS['B']/'TRAINING_CONFIGURATION_AND_SEEDS.json')
    assert {k:v for k,v in a.items() if k not in ['arm','burial_token_enabled']}=={k:v for k,v in b.items() if k not in ['arm','burial_token_enabled']};checks.append('matched configuration apart from arm/token flag')
    assert not a['burial_token_enabled'] and b['burial_token_enabled']
    s=dict(hard_valid=122/128,backbone_valid=122/128,geometry_KL_per_active=.01,structural_pair_RMSD=15.)
    assert safe_initial(s) and safe_monitor(s,s)
    s['backbone_valid']=121/128;assert not safe_initial(s) and not safe_monitor(s,s);checks.append('122/128 passes and121/128 fails unchanged backbone floor')
    p={k:dict(effect=.02,CI=[.01,.03]) for k in ['Bfinal_minus_Afinal','difference_in_changes','B_change','Bfinal_response']}
    assert primary_supported(p);p['difference_in_changes']=dict(effect=0.,CI=[0.,0.]);assert not primary_supported(p);checks.append('insertion-only difference cannot establish token-specific learning')
    assert ci(np.zeros(80))['CI']==[0.,0.] and np.isclose(ci(np.ones(80)*.02)['effect'],.02);checks.append('frozen bootstrap zero/constant contracts')
    module=runpy.run_path(str(ENG/'source96/qualify.py'));module['toy']();checks.append('qualified finite rejected-proposal likelihood and RLOO exact tests reused')
    atomic(PAIR/'qualification/CPU_TESTS.json',dict(PASS=True,checks=checks,VALIDATION_inference=0,production_updates=0),True)
    print('PAIRED_CPU_TESTS_PASS',checks,flush=True)
if __name__=='__main__':main()
