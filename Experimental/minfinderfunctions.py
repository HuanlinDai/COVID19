# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 23:38:36 2020

@author: localaccount
"""

import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

date = datetime.strptime("3/1/20", "%m/%d/%y")
beta1=0.820
beta2=0
gamma=1/14 # 1/(time to recover)
sigma=1/5.1 # 1/(incubation period length)
N=1000 # Population size
initInf = 3
initCond = [N-2.5*initInf,initInf*1.5,initInf,0] #Format: [S,E,I,R]
tEnd1=17
numOfParts=1

tSolveSpace=[0,tEnd1]
tEvalSpace=np.linspace(0,tEnd1,tEnd1-0+1)

# =============================================================================
# DiffEQ
# =============================================================================
def f(t,y): #These are the SEIR equations. They exclude vital dynamics.
    [S,E,I,R]=y
    return [-(beta1*np.exp(beta2*t))*S*I/N, (beta1*np.exp(beta2*t))*S*I/N-sigma*E, sigma*E-gamma*I, gamma*I]
# Solving the IVP
solution = solve_ivp(f, tSolveSpace, initCond, t_eval=tEvalSpace)
tlist = solution.t
ylist = solution.y
betalist = [beta1*np.exp(beta2*n) for n in tlist]
IR = ylist[2]+ylist[3]

DataSample = [3,3,4,5,7,8,8,10,12,15,17,25,37,48,72,100,130,170]
averagesum = sum([abs(IR[n]-DataSample[n]) for n in range(0,17)])

averageslopelist = [abs(IR[n+1]-IR[n]-DataSample[n+1]+DataSample[n]) for n in range(0,17)]
averageslopesum = sum(averageslopelist)
print(averageslopesum)

beta2=0 


def find_min(): #Used for finding the minumum of beta2
# =============================================================================
# DiffEQ
# =============================================================================
    global DataSample
    global averagesum
    global beta4
    beta1=0.820
    gamma=1/14 # 1/(time to recover)
    sigma=1/5.1 # 1/(incubation period length)
    N=1000 # Population size
    initInf = 3
    initCond = [N-2.5*initInf,initInf*1.5,initInf,0] #Format: [S,E,I,R]
    tEnd1=17
    tSolveSpace=[0,tEnd1]
    tEvalSpace=np.linspace(0,tEnd1,tEnd1-0+1)
    def f(t,y): #These are the SEIR equations. They exclude vital dynamics.
        [S,E,I,R]=y
        return [-(beta1*np.exp(beta3*t))*S*I/N, (beta1*np.exp(beta3*t))*S*I/N-sigma*E, sigma*E-gamma*I, gamma*I]
    for i in np.arange(-1,0,0.001):
        beta3=i
        # Solving the IVP
        solution = solve_ivp(f, tSolveSpace, initCond, t_eval=tEvalSpace)
        IR = solution.y[2] + solution.y[3]
        averagesum2 = sum([abs(IR[n]-DataSample[n]) for n in range(0,17)])
        if averagesum2 < averagesum:
            averagesum = averagesum2
        if averagesum2 > averagesum:
            beta4 = beta3-0.001
            break

def find_min_slope():
    beta1=0.820
    gamma=1/14 # 1/(time to recover)
    sigma=1/5.1 # 1/(incubation period length)
    N=1000 # Population size
    initInf = 3
    initCond = [N-2.5*initInf,initInf*1.5,initInf,0] #Format: [S,E,I,R]
    tEnd1=17

    tSolveSpace=[0,tEnd1]
    tEvalSpace=np.linspace(0,tEnd1,tEnd1-0+1)

# =============================================================================
# DiffEQ
# =============================================================================
    def f(t,y): #These are the SEIR equations. They exclude vital dynamics.
        [S,E,I,R]=y
        return [-(beta1*np.exp(-beta3*t))*S*I/N, (beta1*np.exp(-beta3*t))*S*I/N-sigma*E, sigma*E-gamma*I, gamma*I]
    for i in np.arange(0,1,0.0001):
        beta3=i
        # Solving the IVP
        solution = solve_ivp(f, tSolveSpace, initCond, t_eval=tEvalSpace)
        ylist = solution.y
        IR = ylist[2]+ylist[3]
        
        DataSample = [3,3,4,5,7,8,8,10,12,15,17,25,37,48,72,100,130,170]
        averageslopelist = [abs(IR[n+1]-IR[n]-DataSample[n+1]+DataSample[n]) for n in range(0,17)]
        averageslope2 = sum(averageslopelist)
        global averageslopesum
        global beta5
        if averageslope2 < averageslopesum:
            averageslopesum = averageslope2
        if averageslope2 > averageslopesum:
            beta5=beta3
            break
        
find_min()
find_min_slope()
print(averageslopesum)
print(beta5)
beta2=(beta4-4*(beta5-0.0001))/5
solution = solve_ivp(f, tSolveSpace, initCond, t_eval=tEvalSpace)
tlist = solution.t
ylist = solution.y
betalist = [beta1*np.exp(beta2*n) for n in tlist]
irlist = ylist[2]+ylist[3]

tSpaceT=np.linspace(0,tEnd1,tEnd1-0+1)
ypoints=DataSample[:tEnd1+1]

plt.plot(tlist.T, irlist, label="Cumulative Predicted Cases (I+R)")
# Plot formatting
plt.plot(tSpaceT, ypoints, label="Cumulative Reported Cases (I+R)")

plt.legend(loc='best')
axes.set_xlabel('Time since 0 (Days)')
axes.set_ylabel('People')
axes.set_title('COVID19 Model for testing (SEIR, RK4)')
axes.grid()