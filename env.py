import numpy as np


class Env:
    def __init__(self):
        self.T = 30000
        self.t = -1
        self.state = np.zeros(2)
        self.action = np.zeros(2)

        # scenario a
        # self.adv_pert = -0.1 * 1 * np.ones((2, self.T))  # out of max 1, each column is the vector w_t
        # self.adv_co = 1 * np.ones((2, self.T))  # out of max 10, each column is the vector \theta_t^(1)

        # scenario b
        # self.adv_pert = -1 * 1 * np.ones((2, self.T))  # out of max 1, each column is the vector w_t
        # self.adv_co = 10 * np.ones((2, self.T))  # out of max 10, each column is the vector \theta_t^(1)

        #scenario c
        self.adv_pert = -1 * 1 * np.ones((2, self.T))  # out of max 1, each column is the vector w_t
        self.adv_co = 10 * np.ones((2, self.T))  # out of max 10, each column is the vector \theta_t^(1)
        for k in range(0, 15):
            self.adv_co[:, 1000 * (2 * k): 1000 * (2 * k) + 1000] *= -1

    def step(self, action):  # needs t to use w_t
        self.t += 1
        # 1- calculating the reward
        c_t = np.dot(self.adv_co[:, self.t], self.state)
        # 2- moving
        self.state = 0.9 * self.state + action + self.adv_pert[:, self.t]
        # print("new state: ", self.state)
        return c_t.item(), self.adv_co[:, self.t], self.state
