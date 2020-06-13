# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 18:53:22 2020

@author: localaccount
"""
import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
# =============================================================================
# import requests
# import bs4
# =============================================================================

# =============================================================================
# Dates
# =============================================================================
today = datetime.now() # current date and time
yesterday = datetime.strftime(datetime.now() - timedelta(1), "%#m/%#d/%y")

# =============================================================================
# Data from JHU (StartDate = March 1st = day 0)
# =============================================================================
df = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv")
df.drop(df[df['Country_Region']!="US"].index, inplace=True)
StartDate='3/1/20'
PureData = df.loc[:,StartDate:yesterday]
DataSums = PureData.sum(axis=0)
DataList = [DataSums.iloc[n] for n in range(len(DataSums))]

# =============================================================================
# Parameters
# =============================================================================
gamma=1/14 # 1/(time to recover)
sigma=1/5.1 # 1/(incubation period length)
N=329450000 # Population size
initInf = 100
initCond1 = [N-2.5*initInf,initInf*1.5,initInf,0] #Format: [S,E,I,R]
numOfParts=4

times=[0,18,40,65,150]
beta1List=[0.820]
beta2List=[0,-0.0967,-0.0175,0]

for i in range(len(beta2List)):    
    beta1List.append(beta1List[i]*np.exp(beta2List[i]*(times[i+1]-times[i])))

# =============================================================================
# DiffEQ
# =============================================================================
def f(t,y): #These are the SEIR equations. They exclude vital dynamics.
    [S,E,I,R]=y
    return [-(beta1*np.exp(beta2*t))*S*I/N, (beta1*np.exp(beta2*t))*S*I/N-sigma*E, sigma*E-gamma*I, gamma*I]

solutions=[]
tlists=[0]
ylists=[]
allbetalists=[]
initCondList=[initCond1]
z=len(DataList)
for i in range(numOfParts):
    beta1=beta1List[i]
    beta2=beta2List[i]
    newsolution=solve_ivp(f, [0,times[i+1]-times[i]], initCondList[i], t_eval=range(len(np.linspace(times[i],times[i+1],times[i+1]-times[i]+1))))
    initCondList.append([listNums[-1] for listNums in newsolution.y])
    newtlist=[newsolution.t[n] + times[i] for n in range(len(np.linspace(times[i],times[i+1],times[i+1]-times[i]+1)))]
    tlists=np.concatenate((tlists,newtlist[1:]), axis=None)
    if len(ylists)==0:
        ylists=newsolution.y
    else:
        ylists=[np.concatenate((ylists[n],newsolution.y[n][1:]), axis=None) for n in range(len(newsolution.y))]
    solutions.append(newsolution)
    allbetalists+=[beta1List[i]*np.exp(beta2List[i]*n) for n in range(len(newsolution.t))]
    if z>=times[i+1]+1:
        tSpaceT=np.linspace(0,times[i+1],times[i+1]+1)
        ypoints=DataList[:times[i+1]+1]
    else:
        tSpaceT=np.linspace(0,z-1,z)
        ypoints=DataList[:z]
ylistsdf = pd.DataFrame(ylists, index=['S','E','I','R'])
irlist=ylists[2]+ylists[3]

# =============================================================================
# Plots
# =============================================================================
fig, axes = plt.subplots(1, 1, figsize=(20,12))
# Plotting [S, E, I, R]
# =============================================================================
# labels = ["Susceptible", "Exposed", "Infected", "Recovered"]
# for y_arr, label in zip(ylist, labels):
#     if label != "Susceptible":
#         plt.plot(tlist.T, y_arr, label=label)
# 
# =============================================================================
# Plotting Reported Infections

plt.plot(tlists.T, irlist, label="Cumulative Predicted Cases (I+R)")
# Plot formatting
plt.plot(tSpaceT, ypoints, label="Cumulative Reported Cases (I+R)")
plt.legend(loc='best')
axes.set_xlabel('Time since '+StartDate+' (Days)')
axes.set_ylabel('People')
axes.set_title('COVID19 Model for US (SEIR, RK4)')
plt.savefig('CurvesForCOVID19_US_Original.png')
#axes.set_yscale('log')
#plt.savefig('CurvesForCOVID19_US_Logarithmic.png')
# =============================================================================
# Printing
# =============================================================================
print("irlist=")
print(irlist)
#print("betalist=")
#print(betalist)

'''
Sources:
    
    Total Accumulated 
        https://www.cdc.gov/coronavirus/2019-ncov/cases-updates/cases-in-us.html
    Time for incubation:
        https://annals.org/aim/fullarticle/2762808/incubation-period-coronavirus-disease-2019-covid-19-from-publicly-reported
    Google Mobility Reports:
        https://www.google.com/covid19/mobility/
    Data pulled from JHU CSSE:
        https://github.com/CSSEGISandData/COVID-19/blob/master/csse_covid_19_data/csse_covid_19_daily_reports/04-04-2020.csv
    Time for Recovery and Other Statistics:
        https://www.who.int/docs/default-source/coronaviruse/who-china-joint-mission-on-covid-19-final-report.pdf
    SEIR Equations:
        https://www.idmod.org/docs/hiv/model-seir.html
    Date of Stay at Home Mandates by State:
        https://www.nytimes.com/interactive/2020/us/coronavirus-stay-at-home-order.html
    More Data (Unused):
        https://www.worldometers.info/coronavirus/country/us/

'''