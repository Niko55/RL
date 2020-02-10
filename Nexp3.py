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
    # plt.ylim(min(y)-0, max(y) + 20)
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

Regret_vector_per_each_C = []
for c in np.linspace(linspaceparam[0],linspaceparam[1],linspaceparam[2]):
    regret = []
    eta = c * np.sqrt(2 * np.log(K) / (K*T))

    regret.append(eta)

    # Running for given sample_runs for each eta value
    for run in range(0, sample_runs):
        #print ("Eta: ", c, " Run: ", run)

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

        # Each arm  initial loss
        loss_for_each_c = []
        for arm in range(K):
            loss_for_each_c.append(np.exp(-eta * loss_vector_T[0][arm]))
        # print loss_for_each_c
        loss_for_each_c=np.ones(K)
        # Parameter for each run of algorithm
        Cummulative_loss_till_t = np.zeros(K)
        loss_in_sample_Run_till_1_T = 0

        # Computing loss for each value of Regret
        for i in range(0, T):
            # Arm selection has values from 0 to 9 for matching indices
            arm = np.random.choice(np.arange(K), p=weight_of_each_arm)

            loss_in_sample_Run_till_1_T += loss_vector_T[i][arm]

            Cummulative_loss_till_t[arm] += (loss_vector_T[i][arm]*1.0)/weight_of_each_arm[arm]

            loss_for_each_c[arm] = np.exp(-eta * Cummulative_loss_till_t[arm])
            weight_of_each_arm = loss_for_each_c/np.sum(loss_for_each_c)

        regret.append((loss_in_sample_Run_till_1_T - (T * (0.5-delta))))

    Regret_vector_per_each_C.append(regret)

#print (Regret_vector_per_each_C)

 
eta = []
average_regret_per_c = []
Confidence_interval_of_regret_per_C = []
freedom_degree = sample_runs-1
for regret in Regret_vector_per_each_C:
    eta.append(regret[0])
    average_regret_per_c.append(np.mean(regret[1:]))
    Confidence_interval_of_regret_per_C.append(ss.t.ppf(0.95, freedom_degree)*ss.sem(regret[1:]))

plot_errorbar(eta, average_regret_per_c, Confidence_interval_of_regret_per_C, "Regret vs Eta in EXP3", "Eta", "Regret", "EXP3.png")