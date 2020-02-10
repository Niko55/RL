import numpy as np
from scipy.optimize import bisect
from scipy.stats import entropy
import matplotlib.pyplot as plt
import scipy.stats as ss
# Parameters of algorithm

#############################
epsilon = 0.1
delta = 0.1
alpha = 2
K_array = [10, 15, 20, 25, 30]
t = 1
B = np.inf
sampleSize = 200
k_1 = 4*np.e + 4
#############################




#############################
def is_Uba_Lba_possible (bound, p_hat, Na, beta):
    return Na*entropy([p_hat, 1-p_hat],qk=[bound, 1-bound])-beta


def Find_Uba_Lba(p_hat, Na, beta):
    try:
        ub_a = bisect(is_Uba_Lba_possible , p_hat, 1,
                             args=(p_hat, Na, beta))
    except(ValueError):
        ub_a = 1
    try:
        lb_a = bisect(is_Uba_Lba_possible , 0, p_hat,
                             args=(p_hat, Na, beta))
    except(ValueError):
        lb_a = 0

    return ub_a, lb_a

#############################






#############################
arrayOfNumberOfSamplesForEveryK = []
for K in K_array:
    arrayOfNumberOfSamplesForK = [K]
    mistakes = 0
    actual_arm_mean  = [0.5]
    for arm in range(1, K):
        actual_arm_mean.append((0.5) - ((arm+1) / 70.0))

        
    for sample in range(0, sampleSize):


        beta = np.log((k_1 * K * t ** alpha) / delta) + np.log(np.log((k_1 * K * t ** alpha) / delta))

        CummulativeReward_t = np.zeros(K)
        Ub_array = np.zeros(K)
        Lb_array = np.zeros(K)

        for arm in range(K):
            CummulativeReward_t[arm] += np.random.binomial(1, actual_arm_mean [arm], size=None)
            Ub_array[arm], Lb_array[arm] = Find_Uba_Lba(CummulativeReward_t[arm], 1, beta)

        bestArm = np.argmax(CummulativeReward_t)
        Ub_array[bestArm] = 0
        secondBestArm = np.argmax(Ub_array)
        Ubt = Ub_array[secondBestArm]
        Lbt = Lb_array[bestArm]
        B = Ubt - Lbt

        countArm = np.ones(K)
        while B > epsilon:

            t += 1
            beta = np.log((k_1 * K * t ** alpha) / delta) + np.log(np.log((k_1 * K * t ** alpha) / delta))

            CummulativeReward_t[bestArm] += np.random.binomial(1, actual_arm_mean [bestArm], size=None)
            CummulativeReward_t[secondBestArm] += np.random.binomial(
                1, actual_arm_mean [secondBestArm], size=None)
            countArm[bestArm] += 1
            countArm[secondBestArm] += 1

            Ub_array = np.zeros(K)
            Lb_array = np.zeros(K)


            for arm in range(K):
                Ub_array[arm], Lb_array[arm] = Find_Uba_Lba(
                    CummulativeReward_t[arm]/countArm[arm], countArm[arm], beta)

            bestArm = np.argmax(CummulativeReward_t/countArm * 1.0)
            Ub_array[bestArm] = 0
            secondBestArm = np.argmax(Ub_array)
            Ubt = Ub_array[secondBestArm]
            Lbt = Lb_array[bestArm]
            B = Ubt - Lbt
            # print B

        arrayOfNumberOfSamplesForK.append(np.sum(countArm))
        if bestArm != 0:
            mistakes += 1

    arrayOfNumberOfSamplesForK.append(mistakes)
    arrayOfNumberOfSamplesForEveryK.append(arrayOfNumberOfSamplesForK)
    print ( K, "in K-array Finished with avg sample complexity",np.mean(arrayOfNumberOfSamplesForK))



#############################







#############################

colors = list("mybr")

def plot_errorbar(x, y, z, title, x_axis, y_axis, file_name):
    plt.errorbar(x, y, z, color=colors.pop())
    plt.scatter(x, y)
    plt.title(title)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.savefig(file_name)
    plt.show()

    plt.close()


def plot_scatter(x, y, title, x_axis, y_axis, file_name):
    plt.plot(x,y,color=colors.pop())
    plt.scatter(x,y,color=colors.pop())
    plt.title(title)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.savefig(file_name)
    plt.show()
    plt.close()


arms = K_array
arm_mean = []
arm_err = []
arms_mistakes = []
sampleSize = len(arms)
freedom_degree = sampleSize-1
for ele in arrayOfNumberOfSamplesForEveryK:
    arm_mean.append(np.mean(ele[1:len(ele)-1]))
    arm_err.append(ss.t.ppf(0.95, freedom_degree)*ss.sem(ele[1:len(ele)-1]))
    arms_mistakes.append(1.0*ele[len(ele)-1]/sampleSize)
plot_errorbar(arms, arm_mean, arm_err, "K vs Sample Complexity", "K", "Sample Complexity", "KL_UCB_Sample_complexity.png")
plot_scatter(arms, arms_mistakes, "Mistakes Probability vs K", "K", "Mistakes Probability", "KL_UCB_Mistake_probablity.png")


#############################
