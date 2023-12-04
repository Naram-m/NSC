import numpy as np
import cvxpy as cp

class AdaFTRL_C():
    def __init__(self, T, p, g, perts):
        self.kappa_M = np.sqrt(100)
        self.pert = perts
        self.M = cp.Variable(40)
        self.M.value = np.zeros(40)

        # projection things
        self.non_projected_sol_parameter = cp.Parameter(40)
        self.objective = cp.Minimize(cp.sum_squares(self.M - self.non_projected_sol_parameter))
        self.prob = cp.Problem(self.objective, [cp.norm(self.M) <= self.kappa_M])

        # FTRL things
        self.grads = []
        self.grads_norms = []

        self.acc_grads = np.zeros(40)
        self.acc_sigma = 1e-4
        self.sigmas = []
        self.acc_sigma_M = np.zeros(40)
        self.Ms = []
        self.acc_h = 0

    def get_action(self, t):
        if t < 10:
            available = t
            concatenated_perturbations = np.zeros(20)
            concatenated_perturbations[-2 * available:] = self.pert[:, 0:available].transpose().flatten()
            u_1 = np.dot(self.M.value[0: 20], concatenated_perturbations)
            u_2 = np.dot(self.M.value[20: 40], concatenated_perturbations)
        else:
            concatenated_perturbations = self.pert[:, t - 10:t].transpose().flatten()
            u_1 = np.dot(self.M.value[0: 20], concatenated_perturbations)
            u_2 = np.dot(self.M.value[20: 40], concatenated_perturbations)
        return np.array([u_1, u_2])

    def grad_step(self, grad):  # grad, acc_co
        # print("np.linalg.norm(grad)>>>>>>>>>",np.linalg.norm(grad))
        # 2 - preparing params for next round
        if not self.Ms:
            self.Ms.append(self.M.value)
        self.grads.append(grad)
        self.grads_norms.append(np.linalg.norm(grad))
        self.acc_grads += self.grads[-1]
        self.calc_sigma()
        self.acc_sigma += self.sigmas[-1]
        self.acc_sigma_M += self.sigmas[-1] * self.Ms[-1]
        unprojected_M = (self.acc_sigma_M - self.acc_grads) / self.acc_sigma
        # projection
        self.non_projected_sol_parameter.value = unprojected_M
        self.prob.solve(warm_start=True)
        print("new Ada FTRLC params: ", np.round(self.M.value, 3))
        self.Ms.append(self.M.value)

    def calc_sigma(self):
        if not self.sigmas:  # no grads yet
            self.sigmas.append(1e-4)
        else:
            # fixed = np.sqrt((1+2 * 15 * 10*0.75)/(2 * self.kappa_M**2))
            fixed = np.sqrt((0.1**2 + 2 * 10 * 10*1*2)/(2 * 0.1**2 * self.kappa_M**2))
            self.prev_h = self.acc_h
            self.acc_h += np.max([self.grads_norms[-1], self.grads_norms[-1] ** 2])

            # print(">>", np.max([self.grads_norms[-1], self.grads_norms[-1] ** 2]))
            # print("adaftrl acc h: ", h)
            # print("adaftrl fixed: ", fixed)
            # print("adaftrl sigma: ", fixed * (np.sqrt(self.acc_h) - np.sqrt(self.prev_h)))
            # print("ada diff", np.sqrt(self.acc_h) - np.sqrt(self.prev_h))
            self.sigmas.append(fixed * (np.sqrt(self.acc_h) - np.sqrt(self.prev_h)))
