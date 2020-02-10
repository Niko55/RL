#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#frtl
import scipy.stats as ss

import numpy as np
import matplotlib.pyplot as plt

a=10**3
b=10**7
c=10**8
d=10
delta=0.02
c1=1


def min_w(cummulative_loss_vector_t_T_for_every_d,t,eta):
    if t==0:
        cummulative_loss_vector_t_T_for_every_d[t]=(loss_vector_T[t])
    else:
        cummulative_loss_vector_t_T_for_every_d[t]=(cummulative_loss_vector_t_T_for_every_d[t-1]+loss_vector_T[t])
#    print(cummulative_loss_vector_t_T_for_every_d[t])
    
    return np.exp(-eta*cummulative_loss_vector_t_T_for_every_d[t])/np.sum(np.exp((-eta*cummulative_loss_vector_t_T_for_every_d[t])))



def plt_errorbar(x, y, z, title, x_axis, y_axis, file_name):
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

#
#def minvector():
##    print("")
#    min_ = np.min(cummulative_loss_vector_t_T_for_every_d[-1])
#   # print(min_)
#    return min_
#
#


def minvector():
#    print("")
    total_loss_for_each_expert=np.zeros(d)
    for i in range(0,d):
        total_loss_for_each_expert[i]=np.sum(loss_vector_T[:,i:i+1])
    min_ = np.min(total_loss_for_each_expert)
    return min_



diff=[]
count=0

avearge_regret_per_c=[]
a=10**6
b=10**7
c=10**8
c1=1
sampel_runs=50

average_regret_per_each_C=[]
for T in range(a,b,c):
    cost_vector_in_each_C=[]
    average_cost_in_each_C=[]
    for c1 in np.linspace(0.2,2.1,11):
        
        print("\n-------------T:",T,"----------------")
        eta=c1*(np.log(d)/(2*T))**0.5
        cost_vector_in_each_C.append([])


        regret_per_each_run_for_c=[]
    
        for runs in range(sampel_runs):
            
    
            loss_vector1_8_T=np.random.binomial(1,0.5,size=(T,8))
            loss_vector_9_T=np.random.binomial(1,0.5-delta,size=(T,1))
            loss_vector_10_T2=np.random.binomial(1,0.5+delta,size=(int(T/2),1))
            loss_vector_10_T=np.random.binomial(1,0.5-2*delta,size=(int(T/2),1))
                    
            loss_vector_T=np.hstack((loss_vector1_8_T,loss_vector_9_T))
            loss_vector_10=np.vstack((loss_vector_10_T2,loss_vector_10_T))
            loss_vector_T=np.hstack((loss_vector_T,loss_vector_10))
    
    
            cummulative_loss_vector_t_T_for_every_d=np.zeros([T,d])
            
    
            w_tilde_T=[]
            total_cost_T=[];
            totalcost=0
            
            for t in range(T):
                
                wt=min_w(cummulative_loss_vector_t_T_for_every_d,t-1,eta)
    #            if t%1==0 and t>0:
    #                print (wt,end =" ")
    #                print(total_cost_T[-1],end =" ")
                if t==0:
                    total_cost_T.append(loss_vector_T[t].dot(wt))
    #            print(loss_vector_T[t,wt])
                else:
                    total_cost_T.append(total_cost_T[t-1]+loss_vector_T[t].dot(wt))
                    
                w_tilde_T.append(wt)
            
                
    #        plt.plot(total_cost_T[100:],label="cost")
    #        plt.legend()
    #        plt.show()
            min_=minvector()
    
          #  print("|||||",totalcost-min_,end=" ")
            regret_per_each_run_for_c.append((total_cost_T[-1]-min_))
            cost_vector_in_each_C[-1].append((total_cost_T[-1]-min_))
    #        print("cosrt",regret_per_each_run_for_c[-1],end=" ")
        average_cost_in_each_C.append(np.mean(np.array(cost_vector_in_each_C[-1])))



c1 = list(np.linspace(0.2,2.1,11))
regret_mean = []
regret_err = []
freedom_degree = sampel_runs-1
eta_index=0
for _ in c1:
  
    regret_err.append(ss.t.ppf(0.95, freedom_degree)*ss.sem(cost_vector_in_each_C[eta_index]))
    eta_index+=1
plt_errorbar(c1, average_cost_in_each_C, regret_err, "Regret vs Eta in FRTL ", "C", "Regret", "FTRL.png")