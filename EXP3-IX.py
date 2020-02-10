#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 16:42:07 2017

@author: 163190012
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 16:38:26 2017

@author: 163190012
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss

linspaceparam=0.1, 2.1, 20
# Scatter Error bar with scatter plot
def plot_errorbar(x, y, z, title, x_axis, y_axis, file_name):
    plt.figure(figsize=(8,8))
    colors = list("rgbcmyk")
    plt.errorbar(x, y, z, color=colors.pop())
    plt.scatter(x, y)
    plt.title(title)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.savefig(file_name)
    plt.show()
    plt.close()

# Parameters for algorithm
delta = 0.1
T = 10**6
T_2 = T/2
K = 10
sample_runs = 50

# Initial probability vector
weight_of_each_arm = np.full(K, 1.0/K)
# print weight_of_each_arm

Regret_in_each_C_vector_per_each_C = []
for c in np.linspace(linspaceparam[0],linspaceparam[1],linspaceparam[2]):
    Regret_in_each_C = []
    eta = c * np.sqrt(2 * np.log(K) / (K*T))
    gamma =  eta / 2.0
    Regret_in_each_C.append(eta)

    # Running for given sample_runs for each eta value
    for run in range(0, sample_runs):

        # Creating loss values
        

        loss_vector1_8_T=np.ones((T,8))*0.5
        loss_vector_9_T=np.ones((T,1))*(0.5-delta)
        loss_vector_10_T2=np.ones((int(T/2),1))*(0.5+delta)
        loss_vector_10_T=np.ones((int(T/2),1))*(0.5-2*delta)
         
        loss_vector_T=np.hstack((loss_vector1_8_T,loss_vector_9_T))
        loss_vector_10=np.vstack((loss_vector_10_T2,loss_vector_10_T))
        loss_vector_T=np.hstack((loss_vector_T,loss_vector_10))
    

        # Finding minimum loss of arm
        # min_loss = min(np.sum(loss_vector_1_10, axis=0))


        # Parameter for each run of algorithm
        expected_loss = 0
        weights = np.ones(K)

        # Computing loss for each value of Regret_in_each_C
        for i in range(0, T):
            # Arm selection has values from 0 to 9 for matching indices
            
            weight_of_each_arm = weights / np.sum(weights)
            arm = np.random.choice(np.arange(K), p=weight_of_each_arm)
            # print arm

            # Updating estimated cumulative expected loss

                
            expected_loss += loss_vector_T[i][arm]

            # print expected_loss
            estimated_loss = (loss_vector_T[i][arm]*1.0)/(weight_of_each_arm[arm]+gamma)
            # print  estimated_loss[arm]

            # New Probability distribution vector
            eta_loss = np.exp(-eta * estimated_loss)
            weights[arm] *= eta_loss
        # Adding Regret_in_each_C, as 9th arm has least mean i.e. 0.4
        Regret_in_each_C.append((expected_loss - (T * (0.5-delta))))

    # print Regret_in_each_C
    Regret_in_each_C_vector_per_each_C.append(Regret_in_each_C)

#print (Regret_in_each_C_vector_per_each_C)


# Plotting Regret_in_each_C vs Eta
eta = np.linspace(linspaceparam[0],linspaceparam[1],linspaceparam[2])
Regret_in_each_C_mean = []
Regret_in_each_C_err = []
freedom_degree = len(Regret_in_each_C_vector_per_each_C[0]) - 2
for Regret_in_each_C in Regret_in_each_C_vector_per_each_C:
    Regret_in_each_C_mean.append(np.mean(Regret_in_each_C[1:]))
    Regret_in_each_C_err.append(ss.t.ppf(0.95, freedom_degree)*ss.sem(Regret_in_each_C[1:]))

plot_errorbar(eta, Regret_in_each_C_mean, Regret_in_each_C_err, "Regret_in_each_C vs Eta in EXP3IX", "C", "Regret_in_each_C", "EXP3IX.png")