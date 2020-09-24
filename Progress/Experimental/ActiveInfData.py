# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 18:16:22 2020

@author: daihu
"""

import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# =============================================================================
# Dates and Data
# =============================================================================
# Manual
Location = "United States"
StartDate='6/9/20'
EndDate='8/9/20'

# Auto
StartDateObj=datetime.strptime(StartDate,'%m/%d/%y')
EndDateObj=datetime.strptime(EndDate,'%m/%d/%y')
tDays=(EndDateObj-StartDateObj).days
yesterday = datetime.strftime(datetime.now() - timedelta(days=1), "%#m/%#d/%y") # Using yesterday's confirmed data to avoid conficts with incoming data
## CSSE Data
df = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv")
df.drop(df[df['Country_Region']!="US"].index, inplace=True)
try:
    PureData = df.loc[:,StartDate:yesterday]
except KeyError:
    yesterday = datetime.strftime(yesterday - timedelta(days=1), "%#m/%#d/%y")
    PureData = df.loc[:,StartDate:yesterday]
TotalCases = PureData.sum(axis=0)
DataSample = TotalCases.loc[StartDate:EndDate]
## Location and COVIDTracking Data
if Location == "United States":
    df.drop(df[df['Country_Region']!="US"].index, inplace=True)
    InfData=pd.read_csv('https://covidtracking.com/api/v1/us/daily.csv')
else:
    df.drop(df[df['Province_State']!=Location].index, inplace=True)
    InfData=pd.read_csv('https://covidtracking.com/api/v1/states/daily.csv')
    us_state_abbrev = {'Alabama': 'AL','Alaska': 'AK','American Samoa': 'AS','Arizona': 'AZ','Arkansas': 'AR','California': 'CA','Colorado': 'CO','Connecticut': 'CT','Delaware': 'DE','District of Columbia': 'DC','Florida': 'FL','Georgia': 'GA','Guam': 'GU','Hawaii': 'HI','Idaho': 'ID','Illinois': 'IL','Indiana': 'IN','Iowa': 'IA','Kansas': 'KS','Kentucky': 'KY','Louisiana': 'LA','Maine': 'ME','Maryland': 'MD','Massachusetts': 'MA','Michigan': 'MI','Minnesota': 'MN','Mississippi': 'MS','Missouri': 'MO','Montana': 'MT','Nebraska': 'NE','Nevada': 'NV','New Hampshire': 'NH','New Jersey': 'NJ','New Mexico': 'NM','New York': 'NY','North Carolina': 'NC','North Dakota': 'ND','Northern Mariana Islands':'MP','Ohio': 'OH','Oklahoma': 'OK','Oregon': 'OR','Pennsylvania': 'PA','Puerto Rico': 'PR','Rhode Island': 'RI','South Carolina': 'SC','South Dakota': 'SD','Tennessee': 'TN','Texas': 'TX','Utah': 'UT','Vermont': 'VT','Virgin Islands': 'VI','Virginia': 'VA','Washington': 'WA','West Virginia': 'WV','Wisconsin': 'WI','Wyoming': 'WY'}
    StateCode = us_state_abbrev[Location]
    InfData.drop(InfData[InfData['state']!=StateCode].index, inplace=True)
###Converting strange date format %Y%m%d to %m/%d/%y
InfData['date']=[datetime.strftime(datetime.strptime(str(InfData['date'].iloc[n]),'%Y%m%d'), "%#m/%#d/%y") for n in range(len(InfData['date']))]
InfData.set_index('date', inplace=True)
###Creating data structures to work with
InfectedData = InfData.loc[EndDate:StartDate]
daterange=[datetime.strftime(EndDateObj - timedelta(days=x),'%#m/%#d/%y') for x in range(tDays)]
reverseddaterange = [d for d in reversed(daterange)]
ActiveInfs=[sum(InfData.loc[date:datetime.strftime(datetime.strptime(date, '%m/%d/%y')-timedelta(days=14),'%#m/%#d/%y')]['positiveIncrease']) for date in daterange]
    # Population Data
pop = pd.read_csv("https://www2.census.gov/programs-surveys/popest/datasets/2010-2019/national/totals/nst-est2019-alldata.csv")
pop.drop(pop[pop['NAME']!=Location].index, inplace=True)

# =============================================================================
# DiffEQ
# =============================================================================
# Coefficients and Constants
gamma=1/14 # 1/(time to recover)
sigma=1/5.1 # 1/(incubation period length)
#N=329450000 # Population size
N = pop.iloc[0]['POPESTIMATE2019'] #Population size
initInf = DataSample[0]
initCond = [N-2.5*initInf,initInf*1.5,initInf,0] #Format: [S,E,I,R]
numOfParts=1

tSolveSpace=[0,tDays]
tEvalSpace=np.linspace(0,tDays,tDays+1)

# SEIR Equations (Excluding vital dynamics)
def f(t,y,beta1,beta2):
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

iteration=0
def bayesfind(data, beta3, beta4, beta3step, beta4step, lastbetaslist, bestdiff, initcond):
    global iteration
    branchbetas=[[beta3+beta3step,beta4],[beta3,beta4+beta4step],[beta3-beta3step,beta4],[beta3,beta4-beta4step],[beta3+beta3step,beta4+beta4step],[beta3+beta3step,beta4-beta4step],[beta3-beta3step,beta4+beta4step],[beta3-beta3step,beta4-beta4step]]
    if lastbetaslist in branchbetas:
        branchbetas.remove(lastbetaslist)
    branchsolns=[solve_ivp(f, tSolveSpace, initcond, t_eval=tEvalSpace,args=n) for n in branchbetas]
    branchIRs=[n.y[2]+n.y[3] for n in branchsolns]
    branchdiffs=[sum([abs(IRs[n]-data[n]) for n in range(0,tDays)]) for IRs in branchIRs]
    mindex=branchdiffs.index(min(branchdiffs))
    if bestdiff==None:
        tempsoln = solve_ivp(f, tSolveSpace, initcond, t_eval=tEvalSpace,args=[beta3,beta4])
        tempIR = tempsoln.y[2] + tempsoln.y[3]
        tempdiff = sum([abs(tempIR[n]-data[n]) for n in range(0,tDays)])
    else:
        tempdiff=bestdiff
    if min(branchdiffs)<tempdiff:
        return bayesfind(data, branchbetas[mindex][0],branchbetas[mindex][1],beta3step,beta4step,[beta3, beta4], min(branchdiffs), initcond)
    else:
        # for displaying intermediate graphs
        showsoln = solve_ivp(f, tSolveSpace, initCond, t_eval=tEvalSpace, args=[beta3,beta4])
        tlist = showsoln.t
        ylist = showsoln.y
        irlist=ylist[2]+ylist[3]
        fig, axes = plt.subplots(1, 1, figsize=(20,12))
        plt.plot(tlist.T, irlist, label="Cumulative Predicted Cases (I+R)")
        plt.plot(tlist.T, DataSample, label="Cumulative Reported Cases (I+R)")
        axes.grid()
        iteration+=1
        axes.set_title('Iteration '+str(iteration))
        plt.show()
        return [beta3, beta4, tempdiff]

def find_min(data,initcond): #Used for finding the minumum of beta2
# =============================================================================
#     for tempinitInf in np.arange(initInf,initInf+1, 5):
#         tempinitCond = [popsize-2.5*tempinitInf,tempinitInf*1.5,tempinitInf,0]
# =============================================================================
        pairsolns=[bayesfind(data, beta3, beta4, .001, .001, None, None, initcond) for beta3 in [gamma, 1] for beta4 in [0, -.05]]
        return Sort_Tuple(pairsolns)[0]



# =============================================================================
# initCond = [327372363.0138199, 99542.96726253489, 300867.6242399966, 1677226.3946775305] 
# bestcoeffs = find_min(DataSample,initCond)
# 
# print("\nbestcoeffs\n",bestcoeffs)
# beta1 = bestcoeffs[0]
# beta2 = bestcoeffs[1]
# 
# bestsoln = solve_ivp(f, tSolveSpace, initCond, t_eval=tEvalSpace, args=[beta1,beta2])
# tlist = bestsoln.t
# ylist = bestsoln.y
# irlist=ylist[2]+ylist[3]
# fig, axes = plt.subplots(1, 1, figsize=(20,12))
# axes.set_xlabel('Date')
# axes.set_ylabel('People')
# axes.set_title('COVID19 Model for US (SEIR, RK4)')
# plt.plot(reverseddates, irlist, label="Cumulative Predicted Cases (I+R)")
# plt.plot(reverseddates, DataSample, label="Cumulative Reported Cases (I+R)")
# axes.grid()
# plt.show()
# =============================================================================

