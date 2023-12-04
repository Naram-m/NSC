import numpy as np
import cvxpy as cp


class GPC():
    def __init__(self, T, p, g, perts):
        self.kappa_M = np.sqrt(100)       # I will deal with M here as a vector (concatenate two rows)
        self.pert = perts
        self.M = np.zeros(40)
        self.M = cp.Variable(40)
        self.M.value = np.zeros(40)
        self.t=-1
        self.g=g

        # OGD stuff, it knows T
        # self.eta = (2 * np.sqrt(self.kappa_M)) / np.sqrt(g * (g + 15 * p ** 2) * T) # this is from agarwal appendix
        self.eta = (2 * self.kappa_M) / np.sqrt(g * (g + 10 * p ** 2) * T) #this is from agarwal appendix
        self.eta = (2 * self.kappa_M * 0.1) / np.sqrt(g * (g * 0.1**2 + 2 * 10 *10*1*2) * T) #this is my naalysis


        #projection stuff
        self.non_projected_sol_parameter = cp.Parameter(40)
        self.objective = cp.Minimize(cp.sum_squares(self.M - self.non_projected_sol_parameter))
        self.prob = cp.Problem(self.objective, [cp.norm(self.M) <= self.kappa_M])

    def get_action(self, t):
        # concatenate last 10 perturbation vectors, the concatenated vector should be 2x10 = 20
        self.t=t
        if t < 10:
            available = t
            concatenated_perturbations = np.zeros(20)
            # a = concatenated_perturbations[-2*available:]
            # b = self.pert[:, 0:available].transpose().flatten()
            concatenated_perturbations[-2 * available:] = self.pert[:, 0:available].transpose().flatten()
            # concatenated_perturbations[-available-1:] = self.pert[:, 0:available].transpose().flatten()
            u_1 = np.dot(self.M.value[0: 20], concatenated_perturbations)
            u_2 = np.dot(self.M.value[20: 40], concatenated_perturbations)
        else:
            concatenated_perturbations = self.pert[:, t - 10:t].transpose().flatten()
            u_1 = np.dot(self.M.value[0: 20], concatenated_perturbations)
            u_2 = np.dot(self.M.value[20: 40], concatenated_perturbations)
        return np.array([u_1, u_2])

    def grad_step(self, grad):  # grad, acc_co. The grad that will be received here is the flattened
        # update the params based on the revealed grad
        # self.eta = (2 * self.kappa_M * 0.1) / np.sqrt(self.g * (self.g * 0.1**2 + 2 * 10 *10*1*2) * self.t) #this is from agarwal appendix

        unprojected_M = self.M.value - self.eta * grad
        self.non_projected_sol_parameter.value = unprojected_M
        self.prob.solve(warm_start=True)
        print("new params GPC: ", np.round(self.M.value, 3))
        # print(np.sum(np.abs(self.M.value)))
