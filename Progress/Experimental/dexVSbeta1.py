# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 00:43:19 2020

@author: daihu
"""

import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.stats import pearsonr
from scipy.stats import spearmanr
# =============================================================================
# Dates and Data (StartDate = March 1st = day 0)
# =============================================================================
today = datetime.now() # current date and time
yesterday = datetime.strftime(datetime.now() - timedelta(1), "%#m/%#d/%y")
df1 = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv")
df1.drop(df1[df1['Country_Region']!="US"].index, inplace=True)
StartDate='3/1/20'
try:
    PureData = df1.loc[:,StartDate:yesterday]
except KeyError:
    today = datetime.strftime(datetime.now() - timedelta(1), "%#m/%#d/%y") # current date and time
    yesterday = datetime.strftime(datetime.now() - timedelta(2), "%#m/%#d/%y")
    PureData = df1.loc[:,StartDate:yesterday]
DataSums = PureData.sum(axis=0)


today = datetime.now() # current date and time
yesterday = datetime.strftime(datetime.now() - timedelta(1), "%#m/%#d/%y")
df = pd.read_csv("https://raw.githubusercontent.com/COVIDExposureIndices/COVIDExposureIndices/master/dex_data/state_dex.csv")
df = df[['date', 'dex']]
dex = []
BetasData = [0.82, 0.7444192172989581, 0.6758048427853578, 0.6135147708696603, 0.5569660799172466, 0.5056295771634948, 0.45902484643323405, 0.41671575232024693, 0.3783063587541277, 0.34343722375971536, 0.3117820357347982, 0.2830445597677735, 0.25695586541835863, 0.23327181001842798, 0.21177075394125539, 0.19225148645824341, 0.17453134277293514, 0.15844449461117627, 0.14384039837046442, 0.1305823863059896, 0.11854638756943287, 0.10761976713177579, 0.09770027172455206, 0.09770027172455206, 0.09600539043521622, 0.09433991154706345, 0.09270332499416714, 0.0910951295591042, 0.08951483271945322, 0.08796195049695577, 0.08643600730929407, 0.08493653582444043, 0.08346307681753297, 0.0820151790302345, 0.08059239903253117, 0.07919430108692842, 0.07782045701500315, 0.07647044606627057, 0.07514385478932617, 0.07384027690522298, 0.07255931318304543, 0.07130057131764181, 0.07006366580947772, 0.06884821784657383, 0.06765385518849176, 0.06648021205233254, 0.06532692900071271, 0.06419365283168384, 0.06308003647056169, 0.06308003647056169, 0.06308003647056169, 0.06308003647056169, 0.06308003647056169, 0.06308003647056169, 0.06308003647056169, 0.06308003647056169, 0.06308003647056169, 0.06308003647056169, 0.06308003647056169, 0.06308003647056169, 0.06308003647056169, 0.06308003647056169, 0.06308003647056169, 0.06308003647056169, 0.06308003647056169, 0.06308003647056169, 0.06308003647056169, 0.06308003647056169, 0.06308003647056169, 0.06308003647056169, 0.06308003647056169, 0.06308003647056169, 0.06308003647056169, 0.06308003647056169, 0.06308003647056169, 0.06308003647056169, 0.06308003647056169, 0.06308003647056169, 0.06308003647056169, 0.06308003647056169,0.06308003647056169, 0.06308003647056169, 0.06308003647056169, 0.06308003647056169, 0.06308003647056169, 0.06308003647056169, 0.07041486139583383, 0.07860256561057026, 0.0877423202728842, 0.09794482797179947, 0.10933366357978641, 0.12204677101704335, 0.12204677101704335, 0.12162035398359287, 0.12119542680099958, 0.12077198426390018, 0.12035002118511827, 0.1199295323956009, 0.11951051274435512, 0.11909295709838497, 0.11867686034262853, 0.1182622173798954, 0.11784902313080405, 0.11743727253371979, 0.11702696054469262, 0.11661808213739559, 0.11621063230306308, 0.11580460605042951, 0.11539999840566824, 0.11499680441233051, 0.1145950191312849, 0.1141946376406567, 0.11379565503576762, 0.11339806642907575, 0.11300186695011573, 0.11260705174543895, 0.11221361597855423]

StarterDate = datetime.strptime(StartDate, '%m/%d/%y')
EnderDate = datetime.strptime(df.iloc[-1]['date'], '%Y-%m-%d')
EndDate = datetime.strftime(EnderDate,'%#m/%#d/%y')
Number_of_Days = (EnderDate-StarterDate).days
Total_Number_of_Days = (datetime.strptime(df.iloc[-1]['date'], '%Y-%m-%d')-datetime.strptime(df.iloc[0]['date'], '%Y-%m-%d')).days
i_Start = (StarterDate-datetime.strptime(df.iloc[0]['date'], '%Y-%m-%d')).days
for i in range(i_Start, Total_Number_of_Days+1):
    dex.append(sum([df.iloc[i+(Total_Number_of_Days+1)*n]['dex'] for n in range(0,50)])/50)
    
for n in range(25, Number_of_Days):
    dex[n]*=0.57
datelist = pd.date_range(StartDate, df.iloc[-1]['date']).to_pydatetime().tolist()
datelist = [element.strftime("%Y-%m-%d") for element in datelist]
DataAvg = pd.DataFrame({})
DataAvg['date'] = datelist[:len(BetasData)]
DataAvg['dex'] = dex[:len(BetasData)]
DataAvg.set_index('date')
DataSample = DataSums.loc[StartDate:EndDate]
DayList = np.arange(0,Number_of_Days)
DataSample.reindex(DayList)
DataSampleDF = pd.DataFrame({'date' : datelist, 'inf': DataSample})
DataSampleDF.set_index('date')

import xlwt

wb = xlwt.Workbook()
ws = wb.add_sheet('Sheet 1')
first_row = 0
for index, item in enumerate(BetasData):
        ws.write(first_row, index, item) 


fig, axes = plt.subplots(1, 1, figsize=(20,12))
plt.plot(DataAvg['dex'])
plt.plot([300*BetasData[n] for n in range(0,len(BetasData))])
axes.grid()