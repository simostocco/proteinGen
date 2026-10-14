from dataclasses import dataclass


@dataclass
class SearchBudget:
    max_proposals: int
    max_completed: int
    proposals: int = 0
    completed: int = 0

    def proposal(self):
        if self.proposals >= self.max_proposals:
            raise ValueError("proposal budget")
        self.proposals += 1

    def complete(self):
        if self.completed >= self.max_completed:
            raise ValueError("completion budget")
        self.completed += 1


def toy_beam(initial, expand, score, steps, width, budget, segment_length=8):
    if min(width, steps, segment_length) < 1:
        raise ValueError("search configuration")
    beam = [initial]
    for _ in range(steps):
        candidates = []
        for state in beam:
            for c in expand(state, segment_length):
                budget.proposal()
                candidates.append(c)
        unique = {repr(c): c for c in candidates}
        beam = sorted(unique.values(), key=score, reverse=True)[:width]
    for c in beam:
        budget.complete()
    return beam
