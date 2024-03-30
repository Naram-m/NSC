import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

costs_gpc = np.load("./gpc.npy", allow_pickle=True)
costs_optimal = np.load("./optimal.npy", allow_pickle=True)
costs_adaftrlc = np.load("./adaftrlc.npy", allow_pickle=True)

costs_gpc_cum = costs_gpc.cumsum()
# costs_optimal_cum = costs_optimal.cumsum()
costs_adaftrlc_cum = costs_adaftrlc.cumsum()

T = len(costs_gpc)
plt.figure(figsize=(6, 5))


############### accum. testing ########################
gpc_regret = (costs_gpc_cum - costs_optimal)/np.arange(1, T+1)
adaftrlc_regret = (costs_adaftrlc_cum - costs_optimal)/np.arange(1, T+1)

########################################################

plt.plot(np.arange(T)/1000, gpc_regret, color='Black', label="GPC", linewidth=2.5, markevery=2500, marker='x',  markersize=15, markeredgecolor='Grey')
plt.plot(np.arange(T)/1000, adaftrlc_regret, color='Blue', label="AdaFTRL-C", linewidth=2.5, markevery=2500, marker='*',  markersize=15, markeredgecolor='Grey')


plt.figtext(0.9, 0, r'$\times 10^3$', ha="right", fontsize=12)

plt.legend(prop={'size': 13.5}, loc=0)
plt.ylabel(r"Average regret $R_T/T$", fontsize=17)


plt.xlabel(r'Horizon $T$', fontsize=17)
plt.yticks(fontsize=16, weight='bold')
plt.xticks(fontsize=16, weight='bold')

plt.savefig("./a.pdf", bbox_inches = 'tight',pad_inches = 0)
# print ("cost gpc: ", np.sum(costs_gpc))
# print ("cost ft: ", np.sum(costs_ftrl_c))
# print ("cost ada: ", np.sum(costs_adaftrl_c))
# print ("cost optftrl-c: ", np.sum(costs_optftrl_c))
# print ("cost optimal: ", np.sum(costs_optimal))

plt.show()