# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 14:32:13 2020

@author: zc
"""

#Using Dex for Beta
import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
from scipy import stats
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy import signal
from sklearn.linear_model import LinearRegression
import math
# =============================================================================
# Dates and Data (StartDate = March 1st = day 0)
# =============================================================================
Location = "United States"
StartDate='3/1/20'
EndDate='7/15/20'

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
Recovered=[DataSample[n]-ActiveInfs[n] for n in range(len(ActiveInfs))]
# Population Data
pop = pd.read_csv("https://www2.census.gov/programs-surveys/popest/datasets/2010-2019/national/totals/nst-est2019-alldata.csv")
pop.drop(pop[pop['NAME']!=Location].index, inplace=True)


# =============================================================================
# Parameters
# =============================================================================
gamma=1/14 # 1/(time to recover)
sigma=1/5.1 # 1/(incubation period length)
N = pop.iloc[0]['POPESTIMATE2019']# Population size
initInf = 100
initCond = [N-2.5*initInf,initInf*1.5,initInf,0] #Format: [S,E,I,R]
#gamma
SeriousCases=InfData.loc[EndDate:StartDate]['hospitalizedCurrently']
SeriousCasesList = SeriousCases.tolist()
SeriousCasesList.reverse()
SeriousRatio=[SeriousCasesList[n]/ActiveInfs[n] for n in range(len(ActiveInfs))]
gammaList = [14+18*r for r in SeriousRatio]
gammaList = [14 if math.isnan(x) else x for x in gammaList]
today = datetime.now() # current date and time
yesterday = datetime.strftime(datetime.now() - timedelta(1), "%#m/%#d/%y")
df1 = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv")
df1.drop(df1[df1['Country_Region']!="US"].index, inplace=True)

def g(t):
    return gammaList[math.floor(t)]

try:
    PureData = df1.loc[:,StartDate:yesterday]
except KeyError:
    today = datetime.strftime(datetime.now() - timedelta(1), "%#m/%#d/%y") # current date and time
    yesterday = datetime.strftime(datetime.now() - timedelta(2), "%#m/%#d/%y")
    PureData = df1.loc[:,StartDate:yesterday]
    
DataSums = PureData.sum(axis=0)
BetasData = [0.82,
 0.7444192172989581,
 0.6758048427853578,
 0.6135147708696603,
 0.5569660799172466,
 0.5056295771634948,
 0.45902484643323405,
 0.41671575232024693,
 0.3783063587541277,
 0.34343722375971536,
 0.3117820357347982,
 0.2830445597677735,
 0.25695586541835863,
 0.23327181001842798,
 0.21177075394125539,
 0.19225148645824341,
 0.17453134277293514,
 0.15844449461117627,
 0.14384039837046442,
 0.1305823863059896,
 0.11854638756943287,
 0.10761976713177579,
 0.09770027172455206,
 0.09770027172455206,
 0.09600539043521622,
 0.09433991154706345,
 0.09270332499416714,
 0.0910951295591042,
 0.08951483271945322,
 0.08796195049695577,
 0.08643600730929407,
 0.08493653582444043,
 0.08346307681753297,
 0.0820151790302345,
 0.08059239903253117,
 0.07919430108692842,
 0.07782045701500315,
 0.07647044606627057,
 0.07514385478932617,
 0.07384027690522298,
 0.07255931318304543,
 0.07130057131764181,
 0.07006366580947772,
 0.06884821784657383,
 0.06765385518849176,
 0.06648021205233254,
 0.06532692900071271,
 0.06419365283168384,
 0.06308003647056169,
 0.06308003647056169,
 0.06289107993766672,
 0.06270268942491572,
 0.06251486323679277,
 0.06232759968286094,
 0.06214089707774696,
 0.061954753741126134,
 0.06176916799770717,
 0.061584138177217126,
 0.061399662614386376,
 0.06121573964893359,
 0.061032367625550865,
 0.06084954489388874,
 0.06066726980854138,
 0.06048554072903181,
 0.060304356019797084,
 0.06012371405017358,
 0.05994361319438237,
 0.05976405183151453,
 0.05958502834551657,
 0.05940654112517593,
 0.059228588564106406,
 0.05905116906073376,
 0.058874281018281245,
 0.0586979228447553,
 0.05852209295293117,
 0.05834678976033864,
 0.05817201168924779,
 0.05799775716665481,
 0.05782402462426781,
 0.05765081249849275,
 0.05747811923041931,
 0.05730594326580693,
 0.05713428305507075,
 0.05696313705326772,
 0.05679250372008268,
 0.05679250372008268,
 0.06435433773620367,
 0.07292301825391927,
 0.08263260532739256,
 0.09363500889961647,
 0.10610236548749892,
 0.12022973131889043,
 0.12022973131889043,
 0.11980966280798824,
 0.1193910619669537,
 0.11897392366792127,
 0.11855824280094156,
 0.11814401427391875,
 0.11773123301254822,
 0.11731989396025434,
 0.11690999207812859,
 0.11650152234486778,
 0.11609447975671257,
 0.11568885932738618,
 0.11528465608803323,
 0.11488186508715906,
 0.11448048139056881,
 0.11408050008130718,
 0.11368191625959817,
 0.11328472504278495,
 0.11288892156527015,
 0.11249450097845623,
 0.11210145845068607,
 0.11170978916718377,
 0.11131948832999573,
 0.11093055115793181,
 0.11054297288650677,
 0.11015674876788197,
 0.1097718740708071,
 0.10938834408056233,
 0.10900615409890047,
 0.10862529944398946,
 0.10824577545035502,
 0.10786757746882349,
 0.10749070086646484,
 0.10711514102653602,
 0.10674089334842427,
 0.10636795324759085,
 0.10599631615551487,
 0.1056259775196373,
 0.10525693280330521,
 0.10488917748571625,
 0.10452270706186313,
 0.1041575170424786,
 0.10379360295398035,
 0.10343096033841626,
 0.10306958475340973,
 0.10270947177210533,
 0.10235061698311455,
 0.10199301599046173,
 0.10163666441353024,
 0.1012815578870088,
 0.10092769206083803,
 0.10057506260015713,
 0.10022366518525078,
 0.09987349551149627,
 0.09952454928931072,
 0.09917682224409854]

today = datetime.now() # current date and time
yesterday = datetime.strftime(datetime.now() - timedelta(1), "%#m/%#d/%y")
df = pd.read_csv("https://raw.githubusercontent.com/COVIDExposureIndices/COVIDExposureIndices/master/dex_data/state_dex.csv")
df = df[['date', 'dex']]
dex = []
dexPast = []
StarterDate = datetime.strptime(StartDate, '%m/%d/%y')
EnderDate = datetime.strptime(df.iloc[-1]['date'], '%Y-%m-%d')
EndDate = datetime.strftime(EnderDate,'%#m/%#d/%y')
Number_of_Days = (EnderDate-StarterDate).days
Total_Number_of_Days = (datetime.strptime(df.iloc[-1]['date'], '%Y-%m-%d')-datetime.strptime(df.iloc[0]['date'], '%Y-%m-%d')).days
i_Start = (StarterDate-datetime.strptime(df.iloc[0]['date'], '%Y-%m-%d')).days
for i in range(i_Start, Total_Number_of_Days+1):
    dex.append(sum([df.iloc[i+(Total_Number_of_Days+1)*n]['dex'] for n in range(0,50)])/50)

for i in range(i_Start-19, i_Start):
    dexPast.append(sum([df.iloc[i+(Total_Number_of_Days+1)*n]['dex'] for n in range(0,50)])/50)

for n in range(25, Number_of_Days):
    dex[n] *= 0.57
dex = dex[:len(BetasData)]

datelist = pd.date_range(StartDate, df.iloc[-1]['date']).to_pydatetime().tolist()
datelist = [element.strftime("%Y-%m-%d") for element in datelist]
DataAvg = pd.DataFrame({})
DataAvg['date'] = datelist[:len(BetasData)]
DataAvg['dex'] = dex[:len(BetasData)]
DataAvg['z_score'] = stats.zscore(DataAvg['dex'])
DataAvg['ewm_alpha_2']=DataAvg['dex'].ewm(alpha=0.2).mean()
DataAvg['ewm_alpha_3']=DataAvg['dex'].ewm(alpha=0.3).mean()
DataAvg['ewm_alpha_9']=DataAvg['dex'].ewm(alpha=0.9).mean()
DataAvg['Savgol_Filter']=signal.savgol_filter(DataAvg['dex'], 101, 3)
DataAvg.set_index('date')
corr,_ = pearsonr(BetasData, DataAvg['dex'])
print("Correlation between Dex Data (corrected for mask wearing) and Beta:") #Dex is mobility data, and beta is the transmission rate
print(corr,_)
corrr,_ = pearsonr(BetasData, DataAvg['ewm_alpha_2'])
print("Correlation between Dex Data (smoothed) and Beta:")
print(corrr,_)
corrrr,_ = pearsonr(BetasData, DataAvg['Savgol_Filter'])
print("Correlation between Dex Data (filtered) and Beta:")
print(corrrr,_)
x = DataAvg['Savgol_Filter'].values.reshape((-1, 1))
y = BetasData
LinReg = LinearRegression().fit(x, y)
print('Equation: ', LinReg.coef_, '*Dex + ', LinReg.intercept_)
fig, axes = plt.subplots(1, 1, figsize=(20,12))
# =============================================================================
# GraphStartDate= '3/1/20'
# GraphEndDate= '6/10/20'
# StartDateObj=datetime.strptime(StartDate,'%m/%d/%y')
# GraphStartDateObj=datetime.strptime(GraphStartDate,'%m/%d/%y')
# GraphEndDateObj=datetime.strptime(GraphEndDate,'%m/%d/%y')
# StartRange = (GraphStartDateObj-StartDateObj).days
# EndRange = (GraphEndDateObj-StartDateObj).days
# =============================================================================
# =============================================================================
# plt.plot(DataAvg['dex'], Label='Dex')
# =============================================================================
# =============================================================================
# plt.plot(DataAvg['ewm_alpha_2'], label="Smoothed Dex")
# =============================================================================
# =============================================================================
# plt.plot(DataAvg['Savgol_Filter'][StartRange:EndRange], label="Filtered Dex")
# =============================================================================
# =============================================================================
# plt.legend(loc='best')
# =============================================================================
# =============================================================================
# plt.plot([300*BetasData[n] for n in range(0,len(BetasData))], label="300*Beta")
# =============================================================================

def beta(t):
    if t<19:
# =============================================================================
#         return LinReg.coef_*DataAvg['Savgol_Filter'][0] + LinReg.intercept_
# =============================================================================
        return 0.82
    else:
        return abs(LinReg.coef_*DataAvg['Savgol_Filter'][math.floor(t)-19] + LinReg.intercept_)
def f(t,y): #These are the SEIR equations. They exclude vital dynamics.
    [S,E,I,R]=y
    return [-(beta(t))*S*I/N, (beta(t))*S*I/N-sigma*E, sigma*E-(1/g(t))*I, (1/g(t))*I]
tSolveSpace=[0,tEnd1]
tEvalSpace=np.linspace(0,tEnd1,tEnd1+1)
solution = solve_ivp(f, tSolveSpace, initCond, t_eval=tEvalSpace)
tlist = solution.t
ylist = solution.y
irlist1=ylist[2]+ylist[3]

plt.plot(tlist.T, irlist1, label="Cumulative Predicted Cases (I+R)")
plt.plot(DataSample.values)
