from .problem import MaxCutProblem
from .solver_qafem import QAFEMSolver

class QAFEM:
    def __init__(self):
        pass

    @classmethod
    def from_couplings(cls, W, **args):
        obj = cls()
        obj.problem = MaxCutProblem(W)
        obj.set_solver(**args)
        return obj

    def set_solver(self, **args):
        self.solver = QAFEMSolver(self.problem, **args)

    def solve(self):
        return self.solver.solve()