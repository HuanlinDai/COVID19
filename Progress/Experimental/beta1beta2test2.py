# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 00:00:12 2020

@author: localaccount
"""

import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

date = datetime.strptime("3/1/20", "%m/%d/%y")
gamma=1/14 # 1/(time to recover)
sigma=1/5.1 # 1/(incubation period length)
N=1000 # Population size
initInf = 3
initCond = [N-2.5*initInf,initInf*1.5,initInf,0] #Format: [S,E,I,R]
tEnd1=17
numOfParts=1

tSolveSpace=[0,tEnd1]
tEvalSpace=np.linspace(0,tEnd1,tEnd1-0+1)

beta1list = np.arange(0,1,0.1)
beta2list = np.arange(-1,0,0.1)
betapairlist = [(x,y) for x in beta1list for y in beta2list]

DataSample = [3,3,4,5,7,8,8,10,12,15,17,25,37,48,72,100,130,170]

# =============================================================================
# DiffEQ
# =============================================================================
def f(t,y,beta1,beta2): # SEIR equations excluding vital dynamics.
    [S,E,I,R]=y
    return [-(beta1*np.exp(beta2*t))*S*I/N, (beta1*np.exp(beta2*t))*S*I/N-sigma*E, sigma*E-gamma*I, gamma*I]

# =============================================================================
# # Solving the IVP
# solution = solve_ivp(f, tSolveSpace, initCond, t_eval=tEvalSpace)
# tlist = solution.t
# ylist = solution.y
# betalist = [beta1*np.exp(beta2*n) for n in tlist]
# IR = ylist[2]+ylist[3]
# avgslopesum = sum([abs(IR[n+1]-IR[n]-DataSample[n+1]+DataSample[n]) for n in range(len(DataSample))])
# print(averageslopesum)
# avgsum = sum([abs(IR[n]-data[n]) for n in range(len(data))])
# =============================================================================

def findbestpair(data, pairlist):
    if not len(pairlist)<3:
        beta3, beta4 = pairlist[round((len(pairlist)/3))-1]
        beta5, beta6 = pairlist[2*round(len(pairlist)/3)-1]
        solution1 = solve_ivp(f, tSolveSpace, initCond, t_eval=tEvalSpace, args=[beta3,beta4])
        solution2 = solve_ivp(f, tSolveSpace, initCond, t_eval=tEvalSpace, args=[beta5,beta6])
        IR1 = solution1.y[2] + solution1.y[3]
        IR2 = solution2.y[2] + solution2.y[3]
        avgsum1 = sum([abs(IR1[n]-data[n]) for n in range(len(data))])
        avgsum2 = sum([abs(IR2[n]-data[n]) for n in range(len(data))])
        if avgsum1 < avgsum2:
            findbestpair(data, pairlist[:2*round(len(pairlist)/3)-1])
        elif avgsum1 > avgsum2:
            findbestpair(data, pairlist[round((len(pairlist)/3)-1):])
    else:
        beta3, beta4 = pairlist[0]
        beta5, beta6 = pairlist[1]
        solution1 = solve_ivp(f, tSolveSpace, initCond, t_eval=tEvalSpace, args=[beta3,beta4])
        solution2 = solve_ivp(f, tSolveSpace, initCond, t_eval=tEvalSpace, args=[beta5,beta6])
        IR1 = solution1.y[2] + solution1.y[3]
        IR2 = solution2.y[2] + solution2.y[3]
        avgsum1 = sum([abs(IR1[n]-data[n]) for n in range(len(data))])
        avgsum2 = sum([abs(IR2[n]-data[n]) for n in range(len(data))])
        if avgsum1 < avgsum2:
            return(beta3,beta4)
        else:
            return(beta5,beta6)

bestpair = findbestpair(DataSample, betapairlist)
beta1=bestpair[0]
beta2=bestpair[1]
solution = solve_ivp(f, tSolveSpace, initCond, t_eval=tEvalSpace, args=[beta1,beta2])
tlist = solution.t
ylist = solution.y
betalist = [beta1*np.exp(beta2*n) for n in tlist]
irlist = ylist[2]+ylist[3]

tSpaceT=np.linspace(0,tEnd1,tEnd1-0+1)
ypoints=DataSample[:tEnd1+1]

fig, axes = plt.subplots(1, 1, figsize=(15,9))
plt.plot(tlist.T, irlist, label="Cumulative Predicted Cases (I+R)")
# Plot formatting
plt.plot(tSpaceT, ypoints, label="Cumulative Reported Cases (I+R)")

plt.legend(loc='best')
axes.set_xlabel('Time since 0 (Days)')
axes.set_ylabel('People')
axes.set_title('COVID19 Model for testing (SEIR, RK4)')