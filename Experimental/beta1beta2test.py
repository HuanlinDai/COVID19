# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 14:36:00 2020

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

averageslopelist = [abs(IR[n+1]-IR[n]-DataSample[n+1]+DataSample[n]) for n in range(0,17)]
averageslopesum = sum(averageslopelist)
print(averageslopesum)


a = np.arange(0,1,0,1)
b = np.arange(-1,0,0,1)
c = [(x,y) for x in a for y in b]


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