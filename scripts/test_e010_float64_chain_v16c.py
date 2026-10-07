import sys
import pytest
import scipy.optimize
attempts=[]
def forbidden(*a,**k):
    attempts.append('optimizer')
    raise AssertionError('Optimizer is forbidden throughout V16C test execution')
class Guard:
    def pytest_collection_finish(self,session):
        scipy.optimize.minimize=forbidden
        for name,module in list(sys.modules.items()):
            if name.startswith('protein_distance_diffusion.training.') and hasattr(module,'minimize'):
                module.minimize=forbidden
args=['-q','tests/test_e010_float64_chain_v16c.py','tests/test_e010_derivative_resolution_v16b.py',
      'tests/test_e010_sequential_teacher_v16a.py','tests/test_e010_sequential_teacher_v16a_acceptance.py',
      'tests/test_e010_no_new_inversion_v9.py','tests/test_e010_direct_correction_v11.py',
      '-k','not sequential_best_feasible_and_frames']
status=pytest.main(args,plugins=[Guard()])
print('actual optimizer invocation attempts:',len(attempts))
assert not attempts
sys.exit(status)
