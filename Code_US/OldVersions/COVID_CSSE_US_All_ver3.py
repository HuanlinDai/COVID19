# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 18:53:22 2020

@author: daihu
"""

# SEIR Model for the US using a dynamic gamma and accounting for the accuracy of active infections


import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import math

# =============================================================================
# Dates and Data
# =============================================================================
# Manual
Location = "United States"
StartDate='3/1/20'
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
# =============================================================================
# InfData['date']=[datetime.strftime(datetime.strptime(str(InfData['date'].iloc[n]),'%Y%m%d'), "%#m/%#d/%y") for n in range(len(InfData['date']))] #Converting strange date format %Y%m%d to %m/%d/%y
# InfData.set_index('date', inplace=True)
# InfectedData = InfData.loc[EndDate:StartDate]
# daterange=[datetime.strftime(EndDateObj - timedelta(days=x),'%#m/%#d/%y') for x in range(tEnd1+1)]
# reverseddaterange = [d for d in reversed(daterange)]
# ActiveInfs=[sum(InfData.loc[date:datetime.strftime(datetime.strptime(date, '%m/%d/%y')-timedelta(days=14),'%#m/%#d/%y')]['positiveIncrease']) for date in daterange]
# ActiveInfs.reverse()
# =============================================================================
# Population Data
pop = pd.read_csv("https://www2.census.gov/programs-surveys/popest/datasets/2010-2019/national/totals/nst-est2019-alldata.csv")
pop.drop(pop[pop['NAME']!=Location].index, inplace=True)
# =============================================================================
# totalIR2=[inf for inf in reversed(InfData['positive'][EndDate:StartDate])]
# =============================================================================

# =============================================================================
# Parameters
# =============================================================================
gamma=1/14 # 1/(time to recover)
sigma=1/5.1 # 1/(incubation period length)
N=329450000 # Population size
initInf = 100
initCond1 = [N-2.5*initInf,initInf*1.5,initInf,0] #Format: [S,E,I,R]

numOfParts=5
times=[0,18,38,42,65,103,112,142,161]
beta1List=[0.820]
beta2List=[0,-0.0967,-0.12,-0.013,0.008,.09,-0.023,-0.045]
gammalist = [18,17.3,14,14,14,14,14,14]
gammalist2 = [15,15,15,15,15,15,15,15]

# =============================================================================
# numOfParts=1
# times=[0,19,30,43,75,106,161]
# beta1List=[0.83]
# beta2List=[0,-0.101,-0.11,-0.01,.02,-0.0035]
# 
# =============================================================================
#gamma
# =============================================================================
# SeriousCases=InfData.loc[EndDate:StartDate]['hospitalizedCurrently']
# SeriousCasesList = SeriousCases.tolist()
# SeriousCasesList.reverse()
# SeriousRatio=[SeriousCasesList[n]/ActiveInfs[n] for n in range(len(ActiveInfs))]
# gammaList = [14+18*r for r in SeriousRatio]
# gammaList = [15 if math.isnan(x) else x for x in gammaList]
# =============================================================================

for i in range(len(beta2List)):    
    beta1List.append(beta1List[i]*np.exp(beta2List[i]*(times[i+1]-times[i])))

# =============================================================================
# DiffEQ
# =============================================================================
def f(t,y,beta1,beta2): #These are the SEIR equations. They exclude vital dynamics.
    [S,E,I,R]=y
    return [-(beta1*np.exp(beta2*t))*S*I/N, (beta1*np.exp(beta2*t))*S*I/N-sigma*E, sigma*E-gamma*I, gamma*I]

tlists=[0]
ylists=[]
allbetalists=[]
initCondList=[initCond1]
z=len(DataSums)
for i in range(numOfParts):
    beta1=beta1List[i]
    beta2=beta2List[i]
# =============================================================================
#     gamma=1/gammaList[i]
# =============================================================================
    newsolution=solve_ivp(f, [0,times[i+1]-times[i]], initCondList[i], t_eval=range(len(np.linspace(times[i],times[i+1],times[i+1]-times[i]+1))), args=[beta1,beta2])
    initCondList.append([listNums[-1] for listNums in newsolution.y])
    newtlist=[newsolution.t[n] + times[i] for n in range(len(np.linspace(times[i],times[i+1],times[i+1]-times[i]+1)))]
    tlists=np.concatenate((tlists,newtlist[1:]), axis=None)
    if len(ylists)==0:
        ylists=newsolution.y
    else:
        ylists=[np.concatenate((ylists[n],newsolution.y[n][1:]), axis=None) for n in range(len(newsolution.y))]
    allbetalists+=[beta1List[i]*np.exp(beta2List[i]*n) for n in range(len(newsolution.t))]
    if z>=times[i+1]+1:
        tSpaceT=np.linspace(0,times[i+1],times[i+1]+1)
        ypoints=DataSums[:times[i+1]+1]
    else:
        tSpaceT=np.linspace(0,z-1,z)
        ypoints=DataSums[:z]
ylistsdf = pd.DataFrame(ylists, index=['S','E','I','R'])

# =============================================================================
# Plots and Prints
# =============================================================================
GraphStartDate= '5/10/20'
GraphEndDate= '6/30/20'
GraphStartDateObj=datetime.strptime(GraphStartDate,'%m/%d/%y')
GraphEndDateObj=datetime.strptime(GraphEndDate,'%m/%d/%y')
StartRange = (GraphStartDateObj-StartDateObj).days
EndRange = (GraphEndDateObj-StartDateObj).days
# =============================================================================
# ActiveInfsRange = range(len(ActiveInfs))
# =============================================================================
fig, axes = plt.subplots(1, 1, figsize=(15,9))
# Plotting [S, E, I, R]
labels = ["pSusceptible", "pExposed", "pInfected", "pRecovered"]
for y_arr, label in zip(ylists, labels):
    if label in ["pInfected", "pRecovered", "pExposed"]:
        plt.plot(tlists.T[StartRange:EndRange], y_arr[StartRange:EndRange], label=label)
plt.plot(tlists.T[StartRange:EndRange], ylists[2][StartRange:EndRange]+ylists[3][StartRange:EndRange], label="Cumulative Predicted Cases (I+R)")
plt.plot(tSpaceT[StartRange:EndRange], ypoints[StartRange:EndRange], label="Cumulative Reported Cases (I+R)")
# =============================================================================
# plt.plot(ActiveInfsRange[StartRange:EndRange], ActiveInfs[StartRange:EndRange], label="Active Infections")
# plt.plot(ActiveInfsRange[StartRange:EndRange], Recovered[StartRange:EndRange], label="Actual Recovered")
# plt.plot(range(times[numOfParts]+1), ActiveInfs[:times[numOfParts]+1], label="Actual Active Infections")
# plt.plot(range(times[numOfParts]+1), totalIR2[:times[numOfParts]+1], label="totalIR2")
#plt.plot(range(times[numOfParts]+1), RecoveredList[:times[numOfParts]+1], label="Actual Recovered")
# plt.plot(range(times[numOfParts]+1), Recovered2[:times[numOfParts]+1], label="Actual Recovered")
# =============================================================================
plt.legend(loc='best')
axes.set_xlabel('Time since ' + StartDate + ' (Days)')
axes.set_ylabel('People')
axes.set_title('COVID19 Model for US 3 (SEIR, RK4)')
axes.grid()
plt.savefig('CurvesForCOVID19_US_3.png')
#axes.set_yscale('log')
#plt.savefig('CurvesForCOVID19_US_3_Log.png')

print("I+R")
print(ylists[2]+ylists[3])

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