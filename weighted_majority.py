#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 20:10:30 2017

@author: 163190012
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 10:48:15 2017

@author: vivek
"""

import numpy as np  
import matplotlib.pyplot as plt
import scipy.stats as ss


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



def minvector():
#    print("")
    total_loss_for_each_expert=np.zeros(d)
    for i in range(0,d):
        total_loss_for_each_expert[i]=np.sum(loss_vector_T[:,i:i+1])
    min_ = np.min(total_loss_for_each_expert)
    return min_

avearge_regret_per_c=[]
a=10**6
b=10**7
c=10**8
c1=1
sampel_runs=50
for T in range(a,b,c):
    cost_vector_in_each_C=[]
    average_cost_in_each_C=[]

    for c11 in np.linspace(0.2,2.1,11):
        c1=c11
        cost_vector_in_each_C.append([])

        for runs in range(sampel_runs):
           # T=10**5
            d=10
            delta=0.1
        
            loss_vector1_8_T=np.random.binomial(1,0.5,size=(T,8))
            loss_vector_9_T=np.random.binomial(1,0.5-delta,size=(T,1))
            loss_vector_10_T2=np.random.binomial(1,0.5+delta,size=(int(T/2),1))
            loss_vector_10_T=np.random.binomial(1,0.5-2*delta,size=(int(T/2),1))
            
            loss_vector_T=np.hstack((loss_vector1_8_T,loss_vector_9_T))
            loss_vector_10=np.vstack((loss_vector_10_T2,loss_vector_10_T))
            loss_vector_T=np.hstack((loss_vector_T,loss_vector_10))
            min_=minvector()
            w_tilde_T=[]
            total_cost_T=[];
            #weighted majority
            totalcost=0
            eta=c1*(2*np.log(d)/T)**0.5
            w_tilde_plus_t=np.ones(10)
            
            w_tilde_0=np.ones(10)
            for t in range(T):
                w_tilde_t=w_tilde_plus_t
                Zt=np.sum(w_tilde_t)
                wt=w_tilde_t/Zt    
                pt=np.random.choice(np.arange(0, 10),p=wt)
                costt=wt.dot(loss_vector_T[t])
                totalcost+=(costt)
                #update
                w_tilde_plus_t=w_tilde_t*(np.exp(-eta*loss_vector_T[t]))
                #print (pt+1,end=' ')
                w_tilde_T.append(w_tilde_t)
                total_cost_T.append(totalcost)
                #vt= 
                
            #print (totalcost-min_)
            
            cost_vector_in_each_C[-1].append((totalcost-min_))
        average_cost_in_each_C.append(np.mean(np.array(cost_vector_in_each_C[-1])))
        
            
    #        plt.plot(total_cost_T,label="cost")
    #        plt.plot(np.arange(T)**0.5,label="bound")
    #        plt.legend()
    #        plt.show()

    

c11 = list(np.linspace(0.2,2.1,11))
regret_mean = []
regret_err = []
freedom_degree = sampel_runs-1
eta_index=0
for _ in c11:
  
    regret_err.append(ss.t.ppf(0.95, freedom_degree)*ss.sem(cost_vector_in_each_C[eta_index]))
    eta_index+=1
plt_errorbar(c11, average_cost_in_each_C, regret_err, "Regret vs Eta in weighted majority", "C", "Regret", "wigheted.png")