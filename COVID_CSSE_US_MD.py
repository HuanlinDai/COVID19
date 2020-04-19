# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 18:53:22 2020

@author: localaccount
"""
import pandas as pd
import numpy as np
import scipy as sc
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# =============================================================================
# Dates
# =============================================================================
today = datetime.now() # current date and time
yesterday=datetime.strftime(datetime.now() - timedelta(1), "%#m/%#d/%y")

# =============================================================================
# Data recorded (beginning from March 12th = index 0)
# =============================================================================
url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv"
df = pd.read_csv(url)
df.drop(df[df['Province_State']!="Maryland"].index, inplace=True)
PureData = df.loc[:,'3/12/20':yesterday]
DataSums=PureData.sum(axis=0)
DataList = [DataSums.iloc[n] for n in range(len(DataSums))]

# =============================================================================
# DiffEQ
# =============================================================================

# Initial Parameters
beta1=0.700 #Beta1
beta2=0
gamma=1/14 # 1/(time to recover)
sigma=1/5.1 # 1/(incubation period length)
N=6083116 # Population size
initInf=18
initCond = [N-2.5*initInf,initInf*1.5,initInf,0] #Format: [S,E,I,R]
tStart1=0
tEnd1=10
numOfParts=2

tSolveSpace=[tStart1,tEnd1]
tEvalSpace=np.linspace(tStart1,tEnd1,tEnd1-tStart1+1)

# Equation
def f(t,y): #These are the SEIR equations. They exclude vital dynamics.
    [S,E,I,R]=y
    return [-(beta1*np.exp(beta2*t))*S*I/N, (beta1*np.exp(beta2*t))*S*I/N-sigma*E, sigma*E-gamma*I, gamma*I]
# Solving the IVP
solution = solve_ivp(f, tSolveSpace, initCond, t_eval=tEvalSpace)
tlist = solution.t
ylist = solution.y
betalist = [beta1*np.exp(beta2*n) for n in tlist]

# Secondary Parameters if 2 parts
if numOfParts>=2:
    beta1*=np.exp(beta2*(tEnd1-tStart1)) #Beta2
    beta2=-.067
    initCond2 = [listNums[-1] for listNums in solution.y] #Format: [S,E,I,R]
    tStart2=tEnd1
    tEnd2=40
    tSolveSpace2=[tStart1,tEnd2]
    tEvalSpace2=np.linspace(tStart2,tEnd2,tEnd2-tStart2+1)

    
    solution2 = solve_ivp(f, tSolveSpace2, initCond2, t_eval=range(len(tEvalSpace2)))
    tlist2= [solution2.t[n] + tEnd1 for n in range(len(tEvalSpace2))]
    tlist=np.concatenate((tlist,tlist2[1:]), axis=None)
    ylist=[np.concatenate((ylist[n],solution2.y[n][1:]), axis=None) for n in range(len(solution2.y))]
    betalist+=[beta1*np.exp(beta2*n) for n in range(len(solution2.t))]

# Tertiary Paramaters if 3 parts
if numOfParts>=3:
    beta1*=np.exp(beta2*(tEnd2-tStart2)) #Beta3
    beta2=-0.27
    initCond3 = [listNums[-1] for listNums in solution2.y] #Format: [S,E,I,R]
    tStart3=tEnd2
    tEnd3=45
    tSolveSpace3=[tStart1,tEnd3]
    tEvalSpace3=np.linspace(tStart3,tEnd3,tEnd3-tStart3+1)
    
    solution3 = solve_ivp(f, tSolveSpace3, initCond3, t_eval=range(len(tEvalSpace3)))
    tlist3= [solution3.t[n] + tEnd2 for n in range(len(tEvalSpace3))]
    tlist=np.concatenate((tlist,tlist3[1:]), axis=None)
    ylist=[np.concatenate((ylist[n],solution3.y[n][1:]), axis=None) for n in range(len(solution3.y))]
    betalist+=[beta1*np.exp(beta2*n) for n in range(len(solution3.t))]


ylistdf = pd.DataFrame(ylist, index=['S','E','I','R'])
# =============================================================================
# print("Number of Active Infections (I):")
# print(ylistdf.iloc[2])
# =============================================================================
irlist=ylist[2]+ylist[3]

# =============================================================================
# Plotting
# =============================================================================
fig, axes = plt.subplots(1, 1, figsize=(10,6))

# Plotting [S, E, I, R]
# =============================================================================
# labels = ["Susceptible", "Exposed", "Infected", "Recovered"]
# for y_arr, label in zip(ylist, labels):
#     if label != "Susceptible":
#         plt.plot(tlist.T, y_arr, label=label)
# =============================================================================
# Plotting Reported Infections
n=len(DataList)
if numOfParts==1:
    if n>=tEnd1+1:
        tSpaceT=np.linspace(tStart1,tEnd1,tEnd1-tStart1+1)
        ypoints=DataList[tStart1:tEnd1+1]
    else:
        tSpaceT=np.linspace(0,n-1,n)
        ypoints=DataList[tStart1:n]
elif numOfParts==2:
    if n>=tEnd2+1:
        tSpaceT=np.linspace(tStart1,tEnd2,tEnd2-tStart1+1)
        ypoints=DataList[tStart1:tEnd2+1]
    else:
        tSpaceT=np.linspace(0,n-1,n)
        ypoints=DataList[tStart1:n]
elif numOfParts==3:
    if n>=tEnd3+1:
        tSpaceT=np.linspace(tStart1,tEnd3,tEnd3-tStart1+1)
        ypoints=DataList[tStart1:tEnd3+1]
    else:
        tSpaceT=np.linspace(0,n-1,n)
        ypoints=DataList[tStart1:n]

plt.plot(tlist.T, irlist, label="Cumulative Predicted Cases (I+R)")
# Plot formatting
plt.plot(tSpaceT, ypoints, label="Cumulative Reported Cases (I+R)")

plt.legend(loc='best')
axes.set_xlabel('Time since March 12th (Days)')
axes.set_ylabel('People')
axes.set_title('COVID19 Model for MD (SEIR, RK4)')
plt.savefig('CurvesForCOVID19_MD.png')

print("irlist=")
print(irlist)
# =============================================================================
# print("betalist=")
# print(betalist)
# =============================================================================
'''
Sources:
    Time for incubation:
        https://annals.org/aim/fullarticle/2762808/incubation-period-coronavirus-disease-2019-covid-19-from-publicly-reported
    Google Mobility Reports:
        https://www.google.com/covid19/mobility/
'''