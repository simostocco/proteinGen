import importlib.util
from pathlib import Path
import sys
import pytest
BASE=Path(__file__).resolve().parents[2]/'scripts/e016_v3'
# source root is parents[2] (tests/<suite>/file -> source)
BASE=Path(__file__).resolve().parents[2]/'scripts/e016_v3'
sys.path.insert(0,str(BASE))
from practical_split import match,band
from mmseqs_stream import Components

@pytest.mark.parametrize('n,expected',[(49,False),(50,True),(51,True)])
def test_exact_identity_threshold(n,expected):assert match(n,100,1,80,1,80,100,100)[0]==expected
@pytest.mark.parametrize('span,expected',[(79,False),(80,True),(81,True)])
def test_full_both_coverage(span,expected):assert match(50,100,1,span,1,80,100,100)[0]==expected
@pytest.mark.parametrize('ni,expected',[(69,False),(70,True),(71,True)])
def test_domain_identity(ni,expected):assert match(ni,100,1,80,20,119,100,400)[1]==expected
@pytest.mark.parametrize('args',[(70,100,1,80,20,119,100,400),(70,100,20,119,1,80,400,100)])
def test_domain_direction_symmetry(args):assert match(*args)==(False,True)
def test_no_short_tag_domain():assert match(22,22,1,22,50,71,22,400)==(False,False)
def test_minimum_span():assert not match(50,50,1,49,1,50,60,400)[1]
def test_no_whole_identity_shortcut():assert match(70,100,1,80,1,80,100,100)==(True,True)
def test_gap_denominator():assert not match(49,100,1,100,1,100,100,100)[0]
def test_boundaries():assert [band(n) for n in (20,64,65,128,129,256,257,384,385,500)]==[0,0,1,1,2,2,3,3,4,4]
def test_component_skip_equivalence():
 edges=[(0,1),(1,2),(0,2),(3,4),(2,4),(5,6)]
 a=Components(range(7));b=Components(range(7))
 for x,y in edges:a.link(x,y)
 for x,y in edges:
  if b.find(x)!=b.find(y):b.link(x,y)
 assert [[a.find(i)==a.find(j) for j in range(7)] for i in range(7)]==[[b.find(i)==b.find(j) for j in range(7)] for i in range(7)]
def test_context_and_shadow_not_automatic():
 c=Components(range(4));c.link(0,1)
 # co-deposition (1,2), shadow bridge (2,3) are metadata, not calls to link.
 assert c.find(0)==c.find(1) and c.find(0)!=c.find(2) and c.find(2)!=c.find(3)
def test_equal_length_shorter_direction_covered():assert match(70,100,1,80,1,80,100,100)==(True,True)

def test_retrieval_filter_is_superset_of_predicates():
 for ql in [20,49,50,64,100,500]:
  for tl in [20,49,50,64,100,500]:
   for qs in [max(1,int(.79*ql)),max(1,int(.8*ql)),ql]:
    for ts in [max(1,int(.79*tl)),max(1,int(.8*tl)),tl]:
     al=max(qs,ts);full,domain=match(al,al,1,qs,1,ts,ql,tl)
     if full or domain:assert qs/ql>=.75 or ts/tl>=.75

def test_allocator_deterministic_under_input_reordering():
 import numpy as np
 from practical_allocate import allocate_vectors
 xs=[(str(i),np.array([1.,float(i%2),float(i%3==0)])) for i in range(100)]
 a,ca=allocate_vectors(xs);b,cb=allocate_vectors(reversed(xs));assert a==b;np.testing.assert_equal(ca,cb)
 assert set(a.values())=={'TRAIN','VALIDATION','TEST'}
 assert ca[:,0].tolist()==[80,10,10]

def test_allocator_does_not_break_an_indivisible_group():
 import numpy as np
 from practical_allocate import allocate_vectors
 a,c=allocate_vectors([('one',np.array([100.,100.]))]);assert a=={'one':'TRAIN'}
 # This is intentionally inadequate, not repaired by fabricating singleton TEST.
 assert c[1:,0].sum()==0

def test_independent_sw_identity_and_insertion():
 from practical_checks import scalar_sw
 a='ACDEFGHIKLMNPQRSTVWY'*3
 d=scalar_sw(a,a);assert d['nident']==d['alnlen']==len(a)
 b=a[:30]+'WW'+a[30:];d=scalar_sw(a,b);assert d['nident']==len(a) and d['alnlen']==len(b)

def test_proposal_cannot_grant_training_admission():
 from protein_joint_sac.data.pdb_split_proposal import PDBSplitProposal
 with pytest.raises(ValueError,match='SCIENTIFIC_REVIEW_PENDING'):PDBSplitProposal.__new__(PDBSplitProposal).training_population()

@pytest.mark.parametrize('meta,want',[
 ({'method':'X-RAY DIFFRACTION'},('X-RAY DIFFRACTION',)),
 ({'experimental_method':'SOLUTION NMR'},('SOLUTION NMR',)),
 ({'methods':['ELECTRON MICROSCOPY']},('ELECTRON MICROSCOPY',)),
 ({'methods':['SOLUTION NMR','X-RAY DIFFRACTION']},('SOLUTION NMR','X-RAY DIFFRACTION')),
 ({},('UNKNOWN',)),
])
def test_authoritative_method_schemas(meta,want):
 from practical_allocate import experimental_methods
 assert experimental_methods(meta)==want

def test_contradictory_methods_fail_closed():
 from practical_allocate import experimental_methods
 with pytest.raises(ValueError,match='contradictory'):experimental_methods({'method':'SOLUTION NMR','methods':['X-RAY DIFFRACTION']})
