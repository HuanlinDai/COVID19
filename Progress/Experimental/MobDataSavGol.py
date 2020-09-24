# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 17:43:09 2020

@author: daihu
"""

import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
from scipy import signal
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import math


StartDate='3/1/20'
EndDate='8/9/20'

# Auto
StartDateObj=datetime.strptime(StartDate,'%m/%d/%y')
EndDateObj=datetime.strptime(EndDate,'%m/%d/%y')
tEnd1=(EndDateObj-StartDateObj).days

BetasData=[0.82,
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

MobData = pd.read_csv("https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv?cachebust=fad0787569d04d8e")
MobData.drop(MobData[MobData['country_region_code']!="US"].index, inplace=True)
MobData.drop(MobData[MobData['sub_region_1'].notnull()].index, inplace=True)
MobData.reset_index(inplace=True)
MobData['date']=[datetime.strftime(datetime.strptime(MobData['date'][d],'%Y-%m-%d'),'%#m/%#d/%y') for d in range(len(MobData))]
MobData.set_index('date',inplace=True)

MobData['savgol_residential']=signal.savgol_filter(MobData['residential_percent_change_from_baseline'], 101, 3)
MobData['savgol_retail/rec']=signal.savgol_filter(MobData['retail_and_recreation_percent_change_from_baseline'], 101, 3)
MobData['savgol_groc/pharm']=signal.savgol_filter(MobData['grocery_and_pharmacy_percent_change_from_baseline'], 101, 3)
MobData['savgol_parks']=signal.savgol_filter(MobData['parks_percent_change_from_baseline'], 101, 3)
MobData['savgol_transit']=signal.savgol_filter(MobData['transit_stations_percent_change_from_baseline'], 101, 3)
MobData['savgol_work']=signal.savgol_filter(MobData['workplaces_percent_change_from_baseline'], 101, 3)
firstderiv=signal.savgol_filter(MobData['parks_percent_change_from_baseline'], 101, 3, deriv=1)
MobData['firstderiv']=[100*n for n in firstderiv]
secondderiv=signal.savgol_filter(MobData['parks_percent_change_from_baseline'], 101, 3,deriv=2)
MobData['secondderiv']=[2500*n for n in secondderiv]

fig, axes = plt.subplots(1, 1, figsize=(15,9))

plt.plot(range(tEnd1+1), MobData['savgol_residential'][StartDate:EndDate], label="Residential_SavGol")
plt.plot(range(tEnd1+1), MobData['savgol_retail/rec'][StartDate:EndDate], label="Retail/Rec_SavGol")
plt.plot(range(tEnd1+1), MobData['savgol_groc/pharm'][StartDate:EndDate], label="Groc/Pharm_SavGol")
plt.plot(range(tEnd1+1), MobData['savgol_parks'][StartDate:EndDate], label="Parks_SavGol")
plt.plot(range(tEnd1+1), MobData['savgol_transit'][StartDate:EndDate], label="Transit_SavGol")
plt.plot(range(tEnd1+1), MobData['savgol_work'][StartDate:EndDate], label="Work_SavGol")
plt.plot(range(tEnd1+1), MobData['firstderiv'][StartDate:EndDate], label="firstderiv")
plt.plot(range(tEnd1+1), MobData['secondderiv'][StartDate:EndDate], label="secondderiv")

#plt.plot([100*BetasData[n] for n in range(len(BetasData))], label="Beta")
    

plt.legend(loc='best')
axes.set_xlabel('Days?')
axes.set_ylabel('Percent Change compared to baseline')
axes.set_title('COVID19 Model for US (SEIR, RK4)')
axes.grid()

