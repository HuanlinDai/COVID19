# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 14:31:19 2020

@author: localaccount
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
DataSample = DataSums.loc["3/1/20":"3/13/20"]

beta1=0.01
beta2=-.1
gamma=1/14 # 1/(time to recover)
sigma=1/5.1 # 1/(incubation period length)
N=329450000 # Population size
initInf = DataSample[0]
initCond = [N-2.5*initInf,initInf*1.5,initInf,0] #Format: [S,E,I,R]
tEnd1=12
numOfParts=1

tSolveSpace=[0,tEnd1]
tEvalSpace=np.linspace(0,tEnd1,tEnd1-0+1)

# =============================================================================
# DiffEQ
# =============================================================================
def f(t,y,beta1,beta2): #These are the SEIR equations. They exclude vital dynamics.
    [S,E,I,R]=y
    return [-(beta1*np.exp(beta2*t))*S*I/N, (beta1*np.exp(beta2*t))*S*I/N-sigma*E, sigma*E-gamma*I, gamma*I]
# Solving the IVP
solution = solve_ivp(f, tSolveSpace, initCond, t_eval=tEvalSpace,args=[beta1,beta2])
ylist = solution.y
IRlist = ylist[2]+ylist[3]

# =============================================================================
# DataSample = [3,3,4,5,7,8,8,10,12,15,17,25,37,48,72,100,130,170]
# =============================================================================

originaldiff = sum([abs(IRlist[n]-DataSample[n]) for n in range(0,tEnd1)])
BetasList = []

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

def find_min(data,popsize): #Used for finding the minumum of beta2
    global originaldiff
    global averagelist
    global BetasList
    global tEnd1
    global tSolveSpace
    global tEvalSpace
    global gamma
    global sigma
    global initInf
    for tempinitInf in np.arange(initInf,3*initInf,5):
        tempinitCond = [popsize-2.5*tempinitInf,tempinitInf*1.5,tempinitInf,0]
        for beta3 in np.arange(0.01,2,0.01):
            placeholddiff=originaldiff
            for beta4 in np.arange(-.1,0,0.001):
                tempsoln = solve_ivp(f, tSolveSpace, tempinitCond, t_eval=tEvalSpace,args=[beta3,beta4])
                tempIR = tempsoln.y[2] + tempsoln.y[3]
                tempdiff = sum([abs(tempIR[n]-data[n]) for n in range(0,tEnd1)])
                if tempdiff < placeholddiff:
                    placeholddiff = tempdiff
                    if beta4>-0.001:
                        BetasList.append([beta3,beta4,placeholddiff,tempinitInf])
                        break
                elif tempdiff > placeholddiff:
                    BetasList.append([beta3,beta4-0.001,placeholddiff,tempinitInf])
                    break
               
find_min(DataSample,N)

SortedBetasList = Sort_Tuple(BetasList)
bestcoeffs = SortedBetasList[0]
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
# =============================================================================
# axes.set_yscale('log')
# =============================================================================
# #plt.savefig('CurvesForCOVID19_US_Logarithmic.png')
# # =============================================================================
# # Printing
# # =============================================================================
# print("irlist=")
# print(irlist)
# #print("betalist=")
# #print(betalist)