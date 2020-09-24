# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 20:31:36 2020

@author: daihu
"""

import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
# =============================================================================
# Dates and Data (StartDate = March 1st = day 0)
# =============================================================================
today = datetime.now() # current date and time
yesterday = datetime.strftime(datetime.now() - timedelta(1), "%#m/%#d/%y")
df = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv")
df.drop(df[df['Country_Region']!="US"].index, inplace=True)
StartDate='3/1/20'
try:
    PureData = df.loc[:,StartDate:yesterday]
except KeyError:
    today = datetime.strftime(datetime.now() - timedelta(1), "%#m/%#d/%y") # current date and time
    yesterday = datetime.strftime(datetime.now() - timedelta(2), "%#m/%#d/%y")
    PureData = df.loc[:,StartDate:yesterday]
DataSums = PureData.sum(axis=0)
DataSample = DataSums.loc["3/19/20":"4/10/20"]

gamma=1/14 # 1/(time to recover)
sigma=1/5.1 # 1/(incubation period length)
N=329450000 # Population size
initInf = DataSample[0]
initCond = [N-2.5*initInf,initInf*1.5,initInf,0] #Format: [S,E,I,R]
tEnd1=22
numOfParts=1

tSolveSpace=[0,tEnd1]
tEvalSpace=np.linspace(0,tEnd1,tEnd1-0+1)

BetasList=[]

# =============================================================================
# DiffEQ
# =============================================================================
def f(t,y,beta1,beta2): #These are the SEIR equations. They exclude vital dynamics.
    [S,E,I,R]=y
    return [-(beta1*np.exp(beta2*t))*S*I/N, (beta1*np.exp(beta2*t))*S*I/N-sigma*E, sigma*E-gamma*I, gamma*I]

def Sort_Tuple(tup):  
    # getting length of list of tuples 
    lst = len(tup)  
    for i in range(0, lst):  
        for j in range(0, lst-i-1):  
            if (tup[j][2] > tup[j + 1][2]):  
                temp = tup[j]  
                tup[j]= tup[j + 1]  
                tup[j + 1]= temp  
    return tup


# For a given beta1, find a beta2 that allows the minumum absolute difference between model and data
def bayesfind(data, beta3, beta4, beta3step, beta4step, lastbetaslist, bestdiff, initcond):
    branchbetas=[[beta3+beta3step,beta4],[beta3,beta4+beta4step],[beta3-beta3step,beta4],[beta3,beta4-beta4step]]
    if lastbetaslist in branchbetas:
        branchbetas.remove(lastbetaslist)
    branchsolns=[solve_ivp(f, tSolveSpace, initcond, t_eval=tEvalSpace,args=n) for n in branchbetas]
    branchIRs=[n.y[2]+n.y[3] for n in branchsolns]
    branchdiffs=[sum([abs(IRs[n]-data[n]) for n in range(0,tEnd1)]) for IRs in branchIRs]
    mindex=branchdiffs.index(min(branchdiffs))
    if bestdiff==None:
        tempsoln = solve_ivp(f, tSolveSpace, initcond, t_eval=tEvalSpace,args=[beta3,beta4])
        tempIR = tempsoln.y[2] + tempsoln.y[3]
        tempdiff = sum([abs(tempIR[n]-data[n]) for n in range(0,tEnd1)])
    else:
        tempdiff=bestdiff
    if min(branchdiffs)<tempdiff:
        return bayesfind(data, branchbetas[mindex][0],branchbetas[mindex][1],beta3step,beta4step,[beta3, beta4], min(branchdiffs), initcond)
    else:
        return [beta3, beta4, tempdiff, initcond[2]]

def find_min(data,popsize): #Used for finding the minumum of beta2
    for tempinitInf in np.arange(initInf,initInf+21, 20):
        tempinitCond = [popsize-2.5*tempinitInf,tempinitInf*1.5,tempinitInf,0]
        pairsolns=[bayesfind(DataSample, beta3, beta4, .02, .0001, None, None, tempinitCond) for beta3 in [gamma, 1] for beta4 in [0, -.1]]
        SortedSolns = Sort_Tuple(pairsolns)
        BetasList.append(SortedSolns[0])
            
find_min(DataSample,N)

SortedBetasList = Sort_Tuple(BetasList)
print("\nbestcoeffs\n",SortedBetasList[0])
beta1 = SortedBetasList[0][0]
beta2 = SortedBetasList[0][1]
initInf = SortedBetasList[0][3]
initCond = [N-2.5*initInf,initInf*1.5,initInf,0]

bestsoln = solve_ivp(f, tSolveSpace, initCond, t_eval=tEvalSpace, args=[beta1,beta2])
tlist = bestsoln.t
ylist = bestsoln.y
irlist=ylist[2]+ylist[3]
fig, axes = plt.subplots(1, 1, figsize=(20,12))
plt.plot(tlist.T, irlist, label="Cumulative Predicted Cases (I+R)")
plt.plot(tlist.T, DataSample, label="Cumulative Reported Cases (I+R)")
plt.show()