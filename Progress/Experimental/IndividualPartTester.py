# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 22:33:41 2020

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
tEnd1=(EndDateObj-StartDateObj).days
today = datetime.now() # current date and time
yesterday = datetime.strftime(datetime.now() - timedelta(days=1), "%#m/%#d/%y")
## CSSE Data
df = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv")
df.drop(df[df['Country_Region']!="US"].index, inplace=True)
try:
    PureData = df.loc[:,StartDate:yesterday]
except KeyError:
    today = datetime.strftime(datetime.now() - timedelta(days=1), "%#m/%#d/%y") # current date and time
    yesterday = datetime.strftime(datetime.now() - timedelta(days=2), "%#m/%#d/%y")
    PureData = df.loc[:,StartDate:yesterday]
DataSums = PureData.sum(axis=0)
DataSample = DataSums.loc[StartDate:EndDate]

    # Location and COVIDTracking Data
if Location == "United States":
    df.drop(df[df['Country_Region']!="US"].index, inplace=True)
    InfData=pd.read_csv('https://covidtracking.com/api/v1/us/daily.csv')
else:
    df.drop(df[df['Province_State']!=Location].index, inplace=True)
    InfData=pd.read_csv('https://covidtracking.com/api/v1/states/daily.csv')
    us_state_abbrev = {'Alabama': 'AL','Alaska': 'AK','American Samoa': 'AS','Arizona': 'AZ','Arkansas': 'AR','California': 'CA','Colorado': 'CO','Connecticut': 'CT','Delaware': 'DE','District of Columbia': 'DC','Florida': 'FL','Georgia': 'GA','Guam': 'GU','Hawaii': 'HI','Idaho': 'ID','Illinois': 'IL','Indiana': 'IN','Iowa': 'IA','Kansas': 'KS','Kentucky': 'KY','Louisiana': 'LA','Maine': 'ME','Maryland': 'MD','Massachusetts': 'MA','Michigan': 'MI','Minnesota': 'MN','Mississippi': 'MS','Missouri': 'MO','Montana': 'MT','Nebraska': 'NE','Nevada': 'NV','New Hampshire': 'NH','New Jersey': 'NJ','New Mexico': 'NM','New York': 'NY','North Carolina': 'NC','North Dakota': 'ND','Northern Mariana Islands':'MP','Ohio': 'OH','Oklahoma': 'OK','Oregon': 'OR','Pennsylvania': 'PA','Puerto Rico': 'PR','Rhode Island': 'RI','South Carolina': 'SC','South Dakota': 'SD','Tennessee': 'TN','Texas': 'TX','Utah': 'UT','Vermont': 'VT','Virgin Islands': 'VI','Virginia': 'VA','Washington': 'WA','West Virginia': 'WV','Wisconsin': 'WI','Wyoming': 'WY'}
    StateCode = us_state_abbrev[Location]
    InfData.drop(InfData[InfData['state']!=StateCode].index, inplace=True)
InfData['date']=[datetime.strftime(datetime.strptime(str(InfData['date'].iloc[n]),'%Y%m%d'), "%#m/%#d/%y") for n in range(len(InfData['date']))] #Converting strange date format %Y%m%d to %m/%d/%y
InfData.set_index('date', inplace=True)
InfectedData = InfData.loc[EndDate:StartDate]
daterange=[datetime.strftime(EndDateObj - timedelta(days=x),'%#m/%#d/%y') for x in range(tEnd1+1)]
reverseddaterange = [d for d in reversed(daterange)]
ActiveInfs=[sum(InfData.loc[date:datetime.strftime(datetime.strptime(date, '%m/%d/%y')-timedelta(days=14),'%#m/%#d/%y')]['positiveIncrease']) for date in daterange]
ActiveInfs.reverse()
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

tSolveSpace=[0,tEnd1]
tEvalSpace=np.linspace(0,tEnd1,tEnd1-0+1)

def f(t,y,beta1,beta2): #These are the SEIR equations. They exclude vital dynamics.
    [S,E,I,R]=y
    return [-(beta1*np.exp(beta2*t))*S*I/N, (beta1*np.exp(beta2*t))*S*I/N-sigma*E, sigma*E-gamma*I, gamma*I]


# =============================================================================
# bestsoln = solve_ivp(f, tSolveSpace, initCond, t_eval=tEvalSpace, args=[beta1,beta2])
# tlist = bestsoln.t
# ylist = bestsoln.y
# 
# irlist=ylist[2]+ylist[3]
# =============================================================================
fig, axes = plt.subplots(1, 1, figsize=(20,12))
axes.set_xlabel('Date')
axes.set_ylabel('People')
axes.set_title('COVID19 Model for US (SEIR, RK4)')
#plt.plot(reverseddaterange, irlist, label="Cumulative Predicted Cases (I+R)")
plt.plot(daterange, ActiveInfs, label="Cumulative Reported Cases (I+R)")
axes.grid()
plt.show()
