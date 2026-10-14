from ..contracts import PlanRef


class NoPlan:
    def sample(self, N, seed):
        return PlanRef()
