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
df.drop(df[df['Province_State']!="New York"].index, inplace=True)
StartDate = '2/24/20'
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
beta1=1
beta2=0.0
gamma=1/14 # 1/(time to recover)
sigma=1/5.1 # 1/(incubation period length)
N=10617423 # Population size
initInf = 15
initCond = [N-2.5*initInf,initInf*1.5,initInf,0] #Format: [S,E,I,R]
tStart1=0
tEnd1=8
numOfParts=4

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
    beta2=-0.05
    initCond2 = [listNums[-1] for listNums in solution.y] #Format: [S,E,I,R]
    tStart2=tEnd1
    tEnd2=30
    tSolveSpace2=[tStart1,tEnd2]
    tEvalSpace2=np.linspace(tStart2,tEnd2,tEnd2-tStart2+1)
    solution2 = solve_ivp(f, tSolveSpace2, initCond2, t_eval=range(len(tEvalSpace2)))
    tlist2 = [solution2.t[n]+tEnd1 for n in range(len(tEvalSpace2))]
    tlist=np.concatenate((tlist,tlist2[1:]), axis=None)
    ylist=[np.concatenate((ylist[n],solution2.y[n][1:]), axis=None) for n in range(len(solution2.y))]
    betalist+=[beta1*np.exp(beta2*n) for n in range(len(solution2.t))]
# If 3 parts, solving third IVP
if numOfParts>=3:
    beta1*=np.exp(beta2*(tEnd2-tStart2)) #Beta3
    beta2=-0.07
    initCond3 = [listNums[-1] for listNums in solution2.y] #Format: [S,E,I,R]
    tStart3=tEnd2
    tEnd3=64
    tSolveSpace3=[tStart1,tEnd3]
    tEvalSpace3=np.linspace(tStart3,tEnd3,tEnd3-tStart3+1)
    
    solution3 = solve_ivp(f, tSolveSpace3, initCond3, t_eval=range(len(tEvalSpace3)))
    tlist3= [solution3.t[n] + tEnd2 for n in range(len(tEvalSpace3))]
    tlist=np.concatenate((tlist,tlist3[1:]), axis=None)
    ylist=[np.concatenate((ylist[n],solution3.y[n][1:]), axis=None) for n in range(len(solution3.y))]
    betalist+=[beta1*np.exp(beta2*n) for n in range(len(solution3.t))]
if numOfParts>=4:
    beta1*=np.exp(beta2*(tEnd3-tStart3)) #Beta4
    beta2=-0.006
    initCond4 = [listNums[-1] for listNums in solution3.y] #Format: [S,E,I,R]
    tStart4=tEnd3
    tEnd4=100
    tSolveSpace4=[tStart1,tEnd4]
    tEvalSpace4=np.linspace(tStart4,tEnd4,tEnd4-tStart4+1)
    
    solution4 = solve_ivp(f, tSolveSpace4, initCond4, t_eval=range(len(tEvalSpace4)))
    tlist4= [solution4.t[n] + tEnd3 for n in range(len(tEvalSpace4))]
    tlist=np.concatenate((tlist,tlist4[1:]), axis=None)
    ylist=[np.concatenate((ylist[n],solution4.y[n][1:]), axis=None) for n in range(len(solution4.y))]
    betalist+=[beta1*np.exp(beta2*n) for n in range(len(solution4.t))]
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
elif numOfParts==4:
    if n>=tEnd4+1:
        tSpaceT=np.linspace(tStart1,tEnd4,tEnd4-tStart1+1)
        ypoints=yvals31[tStart1:tEnd4+1]
    else:
        tSpaceT=np.linspace(0,n-1,n)
        ypoints=yvals31[tStart1:n]
irlist=ylist[2]+ylist[3]
plt.plot(tlist.T, irlist, label="Predicted Infections (Total Confirmed Cases)")
# Plot formatting
plt.plot(tSpaceT, ypoints, label="Reported Infections (Total Confirmed Cases)")
plt.legend(loc='best')
axes.set_xlabel('Time since ' +StartDate+ ' (Days)')
axes.set_ylabel('People')
axes.set_title('COVID19 Modeling Using SEIR Model and RK4')
plt.savefig('CurvesForCOVID19.png')
'''
Sources:
    Time for incubation:
    https://annals.org/aim/fullarticle/2762808/incubation-period-coronavirus-disease-2019-covid-19-from-publicly-reported
'''
