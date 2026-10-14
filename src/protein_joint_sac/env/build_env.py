from ..contracts import BuildState, GeneratedProtein
from ..geometry.ca_builder import append


class BuildEnv:
    def __init__(self, N):
        self.state = BuildState.empty(N)

    def step(self, action):
        self.state = append(self.state, action)
        return self.state, self.state.t == self.state.N

    def completed(self):
        return GeneratedProtein(self.state)
