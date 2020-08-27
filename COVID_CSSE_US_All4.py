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
from scipy import signal
# =============================================================================
# Dates
# =============================================================================
# Manual
Location = "United States"
StartDate='3/1/20'

# Auto
StartDateObj=datetime.strptime(StartDate,'%m/%d/%y')

todayObj = datetime.now()
yesterdayObj = datetime.now() - timedelta(days=2)
today = datetime.strftime(todayObj, "%#m/%#d/%y")
yesterday = datetime.strftime(yesterdayObj, "%#m/%#d/%y")
tEnd=(todayObj-StartDateObj).days

# =============================================================================
# Data
# =============================================================================
# Cumulative Infections Data (CSSE) and New Cases Data (COVIDTrackingProject)
CSSE = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv")
if Location == "United States":
    CSSE.drop(CSSE[CSSE['Country_Region']!="US"].index, inplace=True)
    CovTrac=pd.read_csv('https://covidtracking.com/api/v1/us/daily.csv')
else:
    CSSE.drop(CSSE[CSSE['Province_State']!=Location].index, inplace=True)
    CovTrac=pd.read_csv('https://covidtracking.com/api/v1/states/daily.csv')
    us_state_abbrev = {'Alabama': 'AL','Alaska': 'AK','American Samoa': 'AS','Arizona': 'AZ','Arkansas': 'AR','California': 'CA','Colorado': 'CO','Connecticut': 'CT','Delaware': 'DE','District of Columbia': 'DC','Florida': 'FL','Georgia': 'GA','Guam': 'GU','Hawaii': 'HI','Idaho': 'ID','Illinois': 'IL','Indiana': 'IN','Iowa': 'IA','Kansas': 'KS','Kentucky': 'KY','Louisiana': 'LA','Maine': 'ME','Maryland': 'MD','Massachusetts': 'MA','Michigan': 'MI','Minnesota': 'MN','Mississippi': 'MS','Missouri': 'MO','Montana': 'MT','Nebraska': 'NE','Nevada': 'NV','New Hampshire': 'NH','New Jersey': 'NJ','New Mexico': 'NM','New York': 'NY','North Carolina': 'NC','North Dakota': 'ND','Northern Mariana Islands':'MP','Ohio': 'OH','Oklahoma': 'OK','Oregon': 'OR','Pennsylvania': 'PA','Puerto Rico': 'PR','Rhode Island': 'RI','South Carolina': 'SC','South Dakota': 'SD','Tennessee': 'TN','Texas': 'TX','Utah': 'UT','Vermont': 'VT','Virgin Islands': 'VI','Virginia': 'VA','Washington': 'WA','West Virginia': 'WV','Wisconsin': 'WI','Wyoming': 'WY'}
    StateCode = us_state_abbrev[Location]
    CovTrac.drop(CovTrac[CovTrac['state']!=StateCode].index, inplace=True)

# Dataframe formatting
CovTrac=CovTrac[::-1] #Reverse CovTrac Data to go from past to present
#Convert strange date format %Y%m%d to %m/%d/%y
CovTrac['date']=[datetime.strftime(datetime.strptime(str(CovTrac['date'].iloc[n]),'%Y%m%d'), "%#m/%#d/%y") for n in range(len(CovTrac['date']))]
CovTrac.set_index('date', inplace=True)
try:
    pureCSSE = CSSE.loc[:,StartDate:today]
    pureCovTrac = CovTrac.loc[StartDate:today]
except KeyError:
    pureCSSE = CSSE.loc[:,StartDate:yesterday]
    pureCovTrac = CovTrac.loc[StartDate:yesterday]
    tEnd-=1

sumCSSE = pureCSSE.sum(axis=0)

daterange=[datetime.strftime(StartDateObj + timedelta(days=x),'%#m/%#d/%y') for x in range(tEnd)]
newInf=[CovTrac.loc[date]['positiveIncrease'] for date in daterange]
savgolnewInf=signal.savgol_filter(newInf,51,5)

# =============================================================================
# Parameters
# =============================================================================
# Population Data
pop = pd.read_csv("https://www2.census.gov/programs-surveys/popest/datasets/2010-2019/national/totals/nst-est2019-alldata.csv")
pop.drop(pop[pop['NAME']!=Location].index, inplace=True)

gamma=1/14 # 1/(time to recover)
sigma=1/5.1 # 1/(incubation period length)
N=329450000 # Population size
initInf = 100
initCond1 = [N-2.5*initInf,initInf*1.5,initInf,0] #Format: [S,E,I,R]

times=[0,18,40,65,100,106,118,123,145,tEnd]
beta1List=[0.82]
beta2List=[0,-0.0967,-0.0175,0,.065,.03,0,-0.02,-0.023]
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
for i in range(len(beta2List)):
    beta1=beta1List[i]
    beta2=beta2List[i]
    ntEval=np.linspace(0,times[i+1]-times[i],times[i+1]-times[i]+1,dtype=int)
    nsoln=solve_ivp(f, [0,ntEval[-1]], initCondList[i], t_eval=ntEval, args=[beta1,beta2])
    initCondList.append([listNums[-1] for listNums in nsoln.y])
    ntlist=[nsoln.t[n] + times[i] for n in range(len(ntEval))]
    tlists=np.concatenate((tlists,ntlist[1:]), axis=None)
    if len(ylists)!=0:
        ylists=[np.concatenate((ylists[n],nsoln.y[n][1:]), axis=None) for n in range(len(nsoln.y))]
    else:
        ylists=nsoln.y
    allbetalists+=[beta1List[i]*np.exp(beta2List[i]*n) for n in range(len(ntEval))]
    if times[i+1]+1>len(sumCSSE):
        tLimit=np.linspace(0,len(sumCSSE),len(sumCSSE)+1)
ylistsdf = pd.DataFrame(ylists, index=['S','E','I','R'])
pnewCases = [ylists[2][n+1]-ylists[2][n]+ylists[3][n+1]-ylists[3][n] for n in range(len(ylists[2])-1)]

# =============================================================================
# Plots and Prints
# =============================================================================
gsd= '3/1/20' # graph start date
ged= '8/25/20' # graph end date
gsdObj=datetime.strptime(gsd,'%m/%d/%y')
gedObj=datetime.strptime(ged,'%m/%d/%y')
rangeStart = (gsdObj-StartDateObj).days
rangeEnd = (gedObj-StartDateObj).days

fig, (ax1,ax2) = plt.subplots(2,figsize=(20,12))

# Plotting [S, E, I, R]
labels = ["Susceptible", "Exposed", "Infected", "Recovered"]
for y_arr, label in zip(ylists, labels):
    if label in ["Infected", "Recovered", "Exposed"]:
        ax1.plot(tlists.T[rangeStart:rangeEnd], y_arr[rangeStart:rangeEnd], label=label)
# Plotting I+R
pIRList=ylists[2]+ylists[3]
ax1.plot(tlists.T[rangeStart:rangeEnd], pIRList[rangeStart:rangeEnd], label="Infected + Recovered")
ax1.plot(tlists.T[rangeStart:rangeEnd], sumCSSE[rangeStart:rangeEnd], label="Reported Cases")
# Plotting New Cases
ax2.plot(daterange[rangeStart:rangeEnd], pnewCases[rangeStart:rangeEnd], label="pNew Cases(dI+dR)")
ax2.plot(daterange[rangeStart:rangeEnd], newInf[rangeStart:rangeEnd], label="Actual New Cases")
ax2.plot(daterange[rangeStart:rangeEnd], savgolnewInf[rangeStart:rangeEnd], label="savgol Actual New Cases")
monthstartnum=[0,15,31,46,61,76,92,107,122,137,153,168]
monthstartdate=[daterange[n] for n in monthstartnum]
for ax in [ax1,ax2]:
    plt.sca(ax)
    plt.xticks(monthstartnum,monthstartdate)

# Plot Config
ax1.legend(loc='best')
ax2.legend(loc='best')
plt.xlabel('Time since ' + StartDate + ' (Days)')
plt.ylabel('People')
ax1.set_title('COVID19 Model 4 for US (SEIR, RK4)')
ax2.set_title('New Cases Per Day')
ax1.grid()
ax2.grid()
# Logarithmic Graph
ax1.set_yscale('log')
plt.savefig('Graphs/CurvesForCOVID19_US_4_Logarithmic.png')
ax1.set_yscale('linear')
plt.savefig('Graphs/CurvesForCOVID19_US_4.png')


print("I+R")
print(ylists[2]+ylists[3])

'''
Sources:
    
    Total Accumulated Cases
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