import numpy as np
from env import Env
from gpc import GPC
from optimal import Optimal
from AdaFTRLC import AdaFTRL_C
np.random.seed(1994)

'''this is the max norm of a vector of length p * size_M, each element can reach up to 10 * 1/delta.
it is the derivative of the cost funciton, which is just [-10 to 10] x.
'''
g = np.linalg.norm(np.ones(10 * 2 * 2) * 10 * (1 / 0.1), 2)  # here it is delta (small),
print("Max g: ", g)

env1 = Env()
env2 = Env()
env3 = Env()
perts = env1.adv_pert
cos = env1.adv_co
T = env1.T

#######################################
grads = []
for t in range(1, T + 1):
    grad = np.zeros((2, 20))  #  10 2x2 matrices
    for j in range(1, 11):  # 10, for each M
        acc = 0
        for i in range(20):  # 20 is how far are we willing to go back in tme
            if t - j - i - 1 < 0:
                break
            else:
                acc += 0.9 ** i * perts[
                    0, t - j - i - 1]  # 0 is the element of the vector w_t, both 0 and 1 are the same.

        # fill the 4 elements with the same grad value
        # print("cos[0, t - 1] * acc>>", cos[0, t - 1] * acc)
        grad[:, (j-1)*2: j*2] = cos[0, t - 1] * acc   # 0 is the element of the vector \theta_t, both 0 and 1 are same.
        # print("GRAD: ", grad)
    grads.append(grad)  # grads is a list, each element is a full 2x20 gradient
    # print ("Grad here: ", grad)
#######################################

gpc1 = GPC(T, 10, g, perts)
gpc1_costs = []
optimal1 = Optimal(T, 10, g, perts=perts)
ada_ftrlc1 = AdaFTRL_C(T, 10, g, perts=perts)

Adaftrl_C1_costs = []

##############################
optimal_costs = []
##############################

costs = 0 * np.ones(T)

for t in range(1, T + 1):
    print("t: ", t)
    # 1- ask GPC for an action
    u_gpc = gpc1.get_action(t)
    u_optimal = optimal1.get_action(t, grads[t-1].flatten())
    # print("grads[t-1].flatten()", grads[t-1].flatten())
    u_AdaFTRL_C = ada_ftrlc1.get_action(t)

    # 2- report it back to env and ask the env about the coefficient and the new state
    (cost, coe, state) = env1.step(u_gpc)
    (cost2, coe2, state2) = env2.step(u_optimal)
    (cost3, coe3, state3) = env3.step(u_AdaFTRL_C)

    # 3- Form the gradient, a vector of length 10
    grad = grads[t - 1]

    # print ("grad",  grad)
    # 5- send the grad to the GPC agent
    gpc1.grad_step(grad.flatten())
    ada_ftrlc1.grad_step(grad.flatten())

    # 6- record the cost
    gpc1_costs.append(cost)
    optimal_costs.append(cost2)
    Adaftrl_C1_costs.append(cost3)


with open('./results3/gpc.npy', 'wb') as f:
    np.save(f, np.array(gpc1_costs, dtype=object))

with open('./results3/adaftrlc.npy', 'wb') as f:
    np.save(f, np.array(Adaftrl_C1_costs, dtype=object))

with open('./results3/optimal.npy', 'wb') as f:
    np.save(f, np.array(optimal_costs, dtype=object))
