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
yesterday=datetime.strftime(datetime.now() - timedelta(1), "%#m/%#d/%y")

# =============================================================================
# Data recorded (beginning from March 10 = index 0)
# =============================================================================
url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv"
df = pd.read_csv(url)
df.drop(df[df['Province_State']!="Georgia"].index, inplace=True)
StartDate = '3/10/20'
PureData = df.loc[:,StartDate:yesterday]
DataSums = PureData.sum(axis=0)
DataList = [DataSums.iloc[n] for n in range(len(DataSums))]
# =============================================================================
# res = requests.get("https://www.worldometers.info/coronavirus/country/us/")
# soup = bs4.BeautifulSoup(res.text,'html.parser')
# datatable=soup.select('tbody')
# =============================================================================

# =============================================================================
# Initial Parameters
# =============================================================================
beta1=0.710
beta2=-0.012
gamma=1/14 # 1/(time to recover)
sigma=1/5.1 # 1/(incubation period length)
N=10617423 # Population size
initInf = 30
initCond = [N-2.5*initInf,initInf*1.5,initInf,0] #Format: [S,E,I,R]
tEnd1=13
numOfParts=3

tSolveSpace=[0,tEnd1]
tEvalSpace=np.linspace(0,tEnd1,tEnd1-0+1)

# =============================================================================
# DiffEQ
# =============================================================================
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
    beta1*=np.exp(beta2*tEnd1) #Beta2
    beta2=-.0843
    initCond2 = [listNums[-1] for listNums in solution.y] #Format: [S,E,I,R]
    tEnd2=48
    tSolveSpace2=[0,tEnd2-tEnd1]
    tEvalSpace2=np.linspace(tEnd1,tEnd2,tEnd2-tEnd1+1)

    solution2 = solve_ivp(f, tSolveSpace2, initCond2, t_eval=range(len(tEvalSpace2)))
    tlist2= [solution2.t[n] + tEnd1 for n in range(len(tEvalSpace2))]
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
    beta2=0
    solution32 = solve_ivp(f, tSolveSpace3, initCond3, t_eval=range(len(tEvalSpace3)))
    ylist2=[np.concatenate((ylist[n],solution32.y[n][1:]), axis=None) for n in range(len(solution32.y))]
    betalist2=betalist+[beta1*np.exp(beta2*n) for n in range(len(solution32.t))]
    #Alternate Sol'n if social distancing is repealed
    x0 = 4
    rho = 5.11
    tox0 = 4
    tEvalSpace32 = np.linspace(tEnd2, tEnd2+tox0, tox0+1)
    # =============================================================================
    # DiffEQ (Gaussian Curve Beta)
    # =============================================================================
    def g(t,y): # Here is a new function that uses a Gaussian curve to model beta.
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

ylistdf = pd.DataFrame(ylist, index=['S','E','I','R'])
# =============================================================================
# print("Number of Active Infections (I):")
# print(ylistdf.iloc[2])
# =============================================================================
irlist1=ylist1[2]+ylist1[3]
irlist2=ylist2[2]+ylist2[3]

# =============================================================================
# Plotting
# =============================================================================
fig, axes = plt.subplots(1, 1, figsize=(20,12))

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
        tSpaceT=np.linspace(0,tEnd1,tEnd1-0+1)
        ypoints=DataList[:tEnd1+1]
    else:
        tSpaceT=np.linspace(0,n-1,n)
        ypoints=DataList[:n]
elif numOfParts==2:
    if n>=tEnd2+1:
        tSpaceT=np.linspace(0,tEnd2,tEnd2-0+1)
        ypoints=DataList[:tEnd2+1]
    else:
        tSpaceT=np.linspace(0,n-1,n)
        ypoints=DataList[:n]
elif numOfParts==3:
    if n>=tEnd3+1:
        tSpaceT=np.linspace(0,tEnd3,tEnd3-0+1)
        ypoints=DataList[:tEnd3+1]
    else:
        tSpaceT=np.linspace(0,n-1,n)
        ypoints=DataList[:n]

plt.plot(tlist.T, irlist1, label="Cumulative Predicted Cases (I+R) w/o repeal")
plt.plot(tlist.T, irlist2, label="Cumulative Predicted Cases (I+R) w/ repeal")
# Plot formatting
plt.plot(tSpaceT, ypoints, label="Cumulative Reported Cases (I+R)")

plt.legend(loc='best')
axes.set_xlabel('Time since '+StartDate+' (Days)')
axes.set_ylabel('People')
axes.set_title('COVID19 Model for US (SEIR, RK4)')
plt.savefig('Graphs/CurvesForCOVID19_GA.png')

# =============================================================================
# Printing
# =============================================================================
print("irlist=")
print(irlist1)
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