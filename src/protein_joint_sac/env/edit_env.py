from dataclasses import dataclass
from ..contracts import EditState, EditAction, BuildState, GeneratedProtein


@dataclass(frozen=True)
class PendingEdit:
    previous: EditState
    protein: GeneratedProtein
    action: EditAction

    def resolve(self, result, old_score, new_score, edit_cost, call_cost):
        state = EditState(
            self.protein,
            result,
            self.previous.remaining_calls - 1,
            self.previous.history + ((self.action.position, self.action.replacement),),
        )
        return state, new_score - old_score - edit_cost - call_cost


def propose(state, action):
    if type(state) != EditState or type(action) != EditAction:
        raise ValueError("separate edit MDP")
    if action.stop:
        return state
    if state.remaining_calls < 1 or action.position >= state.protein.state.N:
        raise ValueError("budget/position")
    if state.protein.sequence[action.position] == action.replacement:
        raise ValueError("no-op rejected")
    residues = list(state.protein.state.residues)
    residues[action.position] = action.replacement
    return PendingEdit(
        state,
        GeneratedProtein(BuildState(tuple(residues), state.protein.state.ca.clone(), state.protein.state.N)),
        action,
    )
