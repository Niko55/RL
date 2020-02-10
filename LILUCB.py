import numpy as np
from scipy.optimize import bisect
from scipy.stats import entropy
import matplotlib.pyplot as plt
import scipy.stats as ss

#############################
epsilon = 0.0+0.0
delta = 0.1
beta = 0.5
sigma = 0.5
K_array = [10, 15, 20, 25, 30]

t = 1
sampleSize = 200
#############################




#############################
arrayOfNumberOfSamplesForEveryK = []
arm_mistakes = []
for K in K_array:

    lambdau = 1 + (10.0/K)

    actual_arm_mean = [0.5]
    for arm in range(1, K):
        actual_arm_mean.append((0.5) - ((arm+1) / 70.0))

    arrayOfNumberOfSamplesForK = [K]
    mistakes = 0

    for instance in range (0, sampleSize):
        CummulativeReward_t = np.zeros(K)
        for arm in range(K):
            CummulativeReward_t[arm] += np.random.binomial(1, actual_arm_mean[arm], size=None)

        countArm = np.ones(K)
        t = K

        flag = True
        while flag:

            t += 1

            confidence = np.zeros(K)
            for i in range(K):
                confidence[i] = (1 + beta) * (1 + np.sqrt(epsilon)) * np.sqrt(
                    (2 * (sigma) ** 2 * (1 + epsilon) * np.log(
                        (np.log(((1 + epsilon) * countArm[i]) + 2)) / delta)) /
                    countArm[i])

            bestArm = np.argmax((CummulativeReward_t/countArm) + confidence)
            CummulativeReward_t[bestArm] += np.random.binomial(1, actual_arm_mean[bestArm], size=None)
            countArm[bestArm] += 1

            for arm in range(0, K):
                if countArm[arm] >= 1 + (lambdau*(np.sum(countArm) - countArm[arm])):
                    flag = False

        bestArm = np.argmax(countArm)
        if bestArm != 0:
            mistakes += 1

        arrayOfNumberOfSamplesForK.append(t)

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
plot_errorbar(arms, arm_mean, arm_err, "Sample Complexity vs K ", "K", "Sample Complexity", "LIL_UCB_Sample_complexity.png")
plot_scatter(arms, arms_mistakes, "Mistakes Probability vs K", "K", "Mistakes Probability", "LIL_UCB_Mistake_probablity.png")


#############################
