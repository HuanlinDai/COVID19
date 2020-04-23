# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 16:41:36 2020

@author: zc
"""

import pandas as pd
import numpy as np
import scipy as sc
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
#import plotly.graph_objects as go
#from tabulate import tabulate

today = datetime.now() # current date and time
yesterday=datetime.strftime(datetime.now() - timedelta(1), "%#m/%#d/%y")

url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv"
df = pd.read_csv(url)
df.drop(df[df['Province_State']!="Georgia"].index, inplace=True)
StartDate = '3/10/20'
PureData = df.loc[:, StartDate:yesterday]
DataSum = PureData.sum(axis=0)
DataList = [DataSum.iloc[n] for n in range(len(DataSum))]


# =============================================================================
# Data recorded (beginning from March 12th = index 0)
# =============================================================================
yvals31 = DataList
# =============================================================================
# Initial Parameters
# =============================================================================

# =============================================================================
# DiffEQ
# =============================================================================
# Equation
beta1=0.71
beta2=-0.012
gamma=1/14 # 1/(time to recover)
sigma=1/5.1 # 1/(incubation period length)
N=10617423 # Population size
initInf = 30
initCond = [N-2.5*initInf,initInf*1.5,initInf,0] #Format: [S,E,I,R]
tStart1=0
tEnd1=13
numOfParts=3

tSolveSpace=[tStart1,tEnd1]
tEvalSpace=np.linspace(tStart1,tEnd1,tEnd1-tStart1+1)

# =============================================================================
# DiffEQ
# =============================================================================
# Equation
def f(t,y): #These are the SEIR equations. They exclude vital dynamics.
    [S,E,I,R]=y
    return [-(beta1*np.exp(beta2*t))*S*I/N, (beta1*np.exp(beta2*t))*S*I/N-sigma*E, sigma*E-gamma*I, gamma*I]
# Solving the IVP
solution = solve_ivp(f, tSolveSpace, initCond, t_eval=tEvalSpace)
tlist = solution.t
ylist = solution.y
betalist = [beta1*np.exp(beta2*n) for n in tlist]
# If 2 parts, solving second IVP
if numOfParts>=2:
# =============================================================================
#     beta1*=np.exp(beta2*(tEnd1-tStart1))
# =============================================================================
    beta1*=np.exp(beta2*(tEnd1-tStart1)) #Beta2
    beta2=-0.0843
    initCond2 = [listNums[-1] for listNums in solution.y] #Format: [S,E,I,R]
    tStart2=tEnd1
    tEnd2=42
    tSolveSpace2=[tStart1,tEnd2]
    tEvalSpace2=np.linspace(tStart2,tEnd2,tEnd2-tStart2+1)
    solution2 = solve_ivp(f, tSolveSpace2, initCond2, t_eval=range(len(tEvalSpace2)))
    tlist2 = [solution2.t[n]+tEnd1 for n in range(len(tEvalSpace2))]
    tlist=np.concatenate((tlist,tlist2[1:]), axis=None)
    ylist=[np.concatenate((ylist[n],solution2.y[n][1:]), axis=None) for n in range(len(solution2.y))]
    betalist+=[beta1*np.exp(beta2*n) for n in range(len(solution2.t))]
# If 3 parts, solving third IVP
if numOfParts>=3:
    beta1*=np.exp(beta2*(tEnd2-tEnd1)) #Beta3
    beta2=-0.0496
    initCond3 = [listNums[-1] for listNums in solution2.y] #Format: [S,E,I,R]
    tEnd3=100
    tSolveSpace3=[0,tEnd3-tEnd2]
    tEvalSpace3=np.linspace(tEnd2,tEnd3,tEnd3-tEnd2+1)
    
    solution31 = solve_ivp(f, tSolveSpace3, initCond3, t_eval=range(len(tEvalSpace3)))
    tlist3= [solution31.t[n] + tEnd2 for n in range(len(tEvalSpace3))]
    tlist=np.concatenate((tlist,tlist3[1:]), axis=None)
    ylist1=[np.concatenate((ylist[n],solution31.y[n][1:]), axis=None) for n in range(len(solution31.y))]
    betalist1=betalist+[beta1*np.exp(beta2*n) for n in range(len(solution31.t))]
    
    #Alternate Sol'n if social distancing is repealed
    x0 = 4
    rho = 5.11
    tox0 = 4
    tEvalSpace32 = np.linspace(tEnd2, tEnd2+tox0, tox0+1)
    #This prediction assumes a gaussian curve rising to a beta value that corresponds to an R0 of 1.1, exactly half of what the data suggests the baseline value was (this is an estimate to account for behavioral changes)
    #The rise time is 4 days given mobility data suggesting a 20% change in mobility for GA requires 4 days
    #The latter half of the beta curve is exponential decay at the same rate as the original model
    # DiffEQ
# =============================================================================
# Equation
    def g(t,y): #These are the SEIR equations. They exclude vital dynamics.
        [S,E,I,R]=y
        return [-(((np.exp((-(t-x0)**2)/50))/(rho*np.pi*2**0.5)))*S*I/N, (((np.exp((-(t-x0)**2)/50))/(rho*np.pi*2**0.5)))*S*I/N-sigma*E, sigma*E-gamma*I, gamma*I]
    
    solution32 = solve_ivp(g, tSolveSpace3, initCond3, t_eval=range(len(tEvalSpace32)))
    ylist2=[np.concatenate((ylist[n],solution32.y[n][1:]), axis=None) for n in range(len(solution32.y))]
    
    initCond32 = [listNums[-1] for listNums in solution32.y] #Format: [S,E,I,R]
    beta1=0.0786
    beta2=-0.012
    tEvalSpace33 = np.linspace(tEnd2+tox0, tEnd3, tEnd3-tEnd2-tox0+1)
    solution33 = solve_ivp(f, tSolveSpace3, initCond32, t_eval=range(len(tEvalSpace33)))
    ylist2=[np.concatenate((ylist2[n],solution33.y[n][1:]), axis=None) for n in range(len(solution33.y))]
    betalist2=betalist+[beta1*np.exp(beta2*n) for n in range(len(solution32.t))]
# =============================================================================
# Plotting
# =============================================================================
fig, axes = plt.subplots(1, 1, figsize=(10,6))

# =============================================================================
# # Plotting [S, E, I, R]
# labels = ["Susceptible", "Exposed", "Infected", "Recovered"]
# for y_arr, label in zip(ylist, labels):
#     if label != "Susceptible":
#         plt.plot(tlist.T, y_arr, label=label)
# =============================================================================

# Plotting Reported Infections
n=len(yvals31)
if numOfParts==1:
    if n>=tEnd1+1:
        tSpaceT=np.linspace(tStart1,tEnd1,tEnd1-tStart1+1)
        ypoints=yvals31[tStart1:tEnd1+1]
    else:
        tSpaceT=np.linspace(0,n-1,n)
        ypoints=yvals31[tStart1:n]
elif numOfParts==2:
    if n>=tEnd2+1:
        tSpaceT=np.linspace(tStart1,tEnd2,tEnd2-tStart1+1)
        ypoints=yvals31[tStart1:tEnd2+1]
    else:
        tSpaceT=np.linspace(0,n-1,n)
        ypoints=yvals31[tStart1:n]
elif numOfParts==3:
    if n>=tEnd3+1:
        tSpaceT=np.linspace(tStart1,tEnd3,tEnd3-tStart1+1)
        ypoints=yvals31[tStart1:tEnd3+1]
    else:
        tSpaceT=np.linspace(0,n-1,n)
        ypoints=yvals31[tStart1:n]
        
irlist1=ylist1[2]+ylist1[3]
irlist2=ylist2[2]+ylist2[3]

plt.plot(tlist.T, irlist1, label="Cumulative Predicted Cases (I+R) w/o repeal")
plt.plot(tlist.T, irlist2, label="Cumulative Predicted Cases (I+R) w/ repeal")
# Plot formatting
plt.plot(tSpaceT, ypoints, label="Cumulative Reported Cases (I+R)")

plt.legend(loc='best')
axes.set_xlabel('Time since '+StartDate+' (Days)')
axes.set_ylabel('People')
axes.set_title('COVID19 Model for US (SEIR, RK4)')
plt.savefig('CurvesForCOVID19_GA.png')

# =============================================================================
# Printing
# =============================================================================
# =============================================================================
# print("irlist=")
# print(irlist1)
# =============================================================================
#print("betalist=")
#print(betalist)
'''
Sources:
    Time for incubation:
    https://annals.org/aim/fullarticle/2762808/incubation-period-coronavirus-disease-2019-covid-19-from-publicly-reported
'''