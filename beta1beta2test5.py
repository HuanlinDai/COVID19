# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 00:24:09 2020

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
DataSample = DataSums.loc["3/1/20":"3/13/20"]

gamma=1/14 # 1/(time to recover)
sigma=1/5.1 # 1/(incubation period length)
N=329450000 # Population size
initInf = DataSample[0]
initCond = [N-2.5*initInf,initInf*1.5,initInf,0] #Format: [S,E,I,R]
tEnd1=12
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

# For a given beta1, find a beta2 that allows the minumum absolute difference between model and data
def twothirdscut(data, beta1, beta2list):
    if len(beta2list)>3:
        threemod2=0    
        beta41 = beta2list[round((len(beta2list)/3))]
        if len(beta2list)%3==2:
            threemod2 = 1
        beta42 = beta2list[2*round((len(beta2list)/3))-threemod2]
        solution1 = solve_ivp(f, tSolveSpace, initCond, t_eval=tEvalSpace, args=[beta1,beta41])
        solution2 = solve_ivp(f, tSolveSpace, initCond, t_eval=tEvalSpace, args=[beta1,beta42])
        IR1 = solution1.y[2] + solution1.y[3]
        IR2 = solution2.y[2] + solution2.y[3]
        avgsum1 = sum([abs(IR1[n]-data[n]) for n in range(len(data))])
        avgsum2 = sum([abs(IR2[n]-data[n]) for n in range(len(data))])
        if avgsum1 < avgsum2:
            return twothirdscut(data, beta1, beta2list[:2*round(len(beta2list)/3)-threemod2])
        else:
            return twothirdscut(data, beta1, beta2list[round((len(beta2list)/3))+1:])
    else:
        beta41, beta42 = beta2list[0], beta2list[1]
        solution1 = solve_ivp(f, tSolveSpace, initCond, t_eval=tEvalSpace, args=[beta1,beta41])
        solution2 = solve_ivp(f, tSolveSpace, initCond, t_eval=tEvalSpace, args=[beta1,beta42])
        IR1 = solution1.y[2] + solution1.y[3]
        IR2 = solution2.y[2] + solution2.y[3]
        avgsum1 = sum([abs(IR1[n]-data[n]) for n in range(len(data))])
        avgsum2 = sum([abs(IR2[n]-data[n]) for n in range(len(data))])
        try:
            beta43 = beta2list[2]
            solution3 = solve_ivp(f, tSolveSpace, initCond, t_eval=tEvalSpace, args=[beta1,beta43])
            IR3 = solution3.y[2] + solution3.y[3]
            avgsum3 = sum([abs(IR3[n]-data[n]) for n in range(len(data))])
            smallest=min([avgsum1, avgsum2, avgsum3])
            if smallest==avgsum1:
                return beta41
            elif smallest==avgsum2:
                return beta42
            else:
                return beta43
        except IndexError:
            if avgsum1<avgsum2:
                return beta41
            else:
                return beta42
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
            bestbeta4 = twothirdscut(DataSample, beta3, np.arange(-.1,0,0.0001))
            tempsoln = solve_ivp(f, tSolveSpace, tempinitCond, t_eval=tEvalSpace,args=[beta3,bestbeta4])
            tempIR = tempsoln.y[2] + tempsoln.y[3]
            tempdiff = sum([abs(tempIR[n]-data[n]) for n in range(0,tEnd1)])
            BetasList.append([beta3,bestbeta4,tempdiff, tempinitInf])
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