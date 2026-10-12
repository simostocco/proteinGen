"""Focused operational hold regression; no models, CUDA or artifacts."""
import ast,json,types,unittest
from pathlib import Path
source=Path(__file__).with_name('supervise_e007_epoch1_hold.py').read_text()
workflow=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='workflow')
early=next(n for n in workflow.body if isinstance(n,ast.If) and "COMMITTED.json" in ast.unparse(n.test))
late=next(n for n in ast.walk(workflow) if isinstance(n,ast.If) and 'request' in ast.unparse(n) and 'EPOCH1_OPERATIONAL_HOLD' in ast.unparse(n.test))
class FakePath:
 def __init__(self,name,files):self.name,self.files=name,files
 def __truediv__(self,name):return FakePath(name,self.files)
 def exists(self):return self.name in self.files
 def read_text(self):return json.dumps(self.files[self.name])
def run_branch(branch,exposures,diagnostics=True):
 files={'EPOCH1_OPERATIONAL_HOLD.json':dict(active=True,target_exposures=230132),'COMMITTED.json':dict(exposures=exposures,epoch=1),'STATE.json':dict(phase='explicit_resume_requested')}
 if diagnostics:files['EPOCH_1_DIAGNOSTICS_COMPLETE.json']={}
 body=[branch,ast.Return(value=ast.Constant('continued'))]
 fn=ast.FunctionDef(name='test',args=ast.arguments(posonlyargs=[],args=[],kwonlyargs=[],kw_defaults=[],defaults=[]),body=body,decorator_list=[])
 scope=dict(RUN=FakePath('root',files),json=json,time=types.SimpleNamespace(time=lambda:1),os=types.SimpleNamespace(getpid=lambda:99),completed=1,atomic_json=lambda p,x:files.__setitem__(p.name,x))
 exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'hold_regression','exec'),scope)
 return scope['test'](),files
class HoldTests(unittest.TestCase):
 def test_exact_boundary_after_diagnostics(self):
  result,files=run_branch(late,230132);self.assertIsNone(result);self.assertEqual(files['STATE.json']['phase'],'paused_epoch1_comparative_evaluation')
 def test_restart_cannot_enter_epoch2(self):
  result,files=run_branch(early,230132);self.assertIsNone(result);self.assertEqual(files['STATE.json']['phase'],'paused_epoch1_comparative_evaluation')
 def test_crossed_boundary_never_trains_more(self):
  result,files=run_branch(early,230133);self.assertIsNone(result);self.assertEqual(files['STATE.json']['phase'],'paused_after_epoch1_boundary_crossed')
 def test_incomplete_training_continues(self):self.assertEqual(run_branch(early,115067)[0],'continued')
 def test_diagnostics_still_required(self):self.assertEqual(run_branch(early,230132,False)[0],'continued')
 def test_wrong_final_count_fails_closed(self):
  with self.assertRaises(AssertionError):run_branch(late,230133)
if __name__=='__main__':unittest.main()
