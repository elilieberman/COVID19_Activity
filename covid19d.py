# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 16:06:13 2020

@author: runner
"""
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 15:24:43 2020

@author: runner
"""
import numpy as np
import pandas as pd
import os
import sys # for utilities that check memory issues
os.environ['PROJ_LIB'] = r'C:\EliPersonal\Python\Datasets\ieee' # fixes cant find "epsg" and PROJ_LIB for BASEMAP
pd.set_option('display.max_columns', None)
pd.option_context('display.max_colwidth', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None) #dont truncate column value display
pd.set_option('display.max_columns', None) # .head will show all
pd.set_option('display.max_rows', None) # .head will show all
#%% https://worldpopulationreview.com/#liveWorldPop population stats
# cov = pd.read_csv(r'C:\EliPersonal\Python\Datasets\Doit\covid.csv', parse_dates = [2], infer_datetime_format = True, index_col = [1,2])
# Import tables
c_usecols = ["Province/State", "Country/Region", "Lat", "Long", "Date","Value", "ISO 3166-1 Alpha 3-Codes"]
confirmed = pd.read_csv('https://data.humdata.org/hxlproxy/data/download/time_series_covid19_confirmed_global_narrow.csv?dest=data_edit&filter01=merge&merge-url01=https%3A%2F%2Fdocs.google.com%2Fspreadsheets%2Fd%2Fe%2F2PACX-1vTglKQRXpkKSErDiWG6ycqEth32MY0reMuVGhaslImLjfuLU0EUgyyu2e-3vKDArjqGX7dXEBV8FJ4f%2Fpub%3Fgid%3D1326629740%26single%3Dtrue%26output%3Dcsv&merge-keys01=%23country%2Bname&merge-tags01=%23country%2Bcode%2C%23region%2Bmain%2Bcode%2C%23region%2Bsub%2Bcode%2C%23region%2Bintermediate%2Bcode&filter02=merge&merge-url02=https%3A%2F%2Fdocs.google.com%2Fspreadsheets%2Fd%2Fe%2F2PACX-1vTglKQRXpkKSErDiWG6ycqEth32MY0reMuVGhaslImLjfuLU0EUgyyu2e-3vKDArjqGX7dXEBV8FJ4f%2Fpub%3Fgid%3D398158223%26single%3Dtrue%26output%3Dcsv&merge-keys02=%23adm1%2Bname&merge-tags02=%23country%2Bcode%2C%23region%2Bmain%2Bcode%2C%23region%2Bsub%2Bcode%2C%23region%2Bintermediate%2Bcode&merge-replace02=on&merge-overwrite02=on&filter03=explode&explode-header-att03=date&explode-value-att03=value&filter04=rename&rename-oldtag04=%23affected%2Bdate&rename-newtag04=%23date&rename-header04=Date&filter05=rename&rename-oldtag05=%23affected%2Bvalue&rename-newtag05=%23affected%2Binfected%2Bvalue%2Bnum&rename-header05=Value&filter06=clean&clean-date-tags06=%23date&filter07=sort&sort-tags07=%23date&sort-reverse07=on&filter08=sort&sort-tags08=%23country%2Bname%2C%23adm1%2Bname&tagger-match-all=on&tagger-default-tag=%23affected%2Blabel&tagger-01-header=province%2Fstate&tagger-01-tag=%23adm1%2Bname&tagger-02-header=country%2Fregion&tagger-02-tag=%23country%2Bname&tagger-03-header=lat&tagger-03-tag=%23geo%2Blat&tagger-04-header=long&tagger-04-tag=%23geo%2Blon&header-row=1&url=https%3A%2F%2Fraw.githubusercontent.com%2FCSSEGISandData%2FCOVID-19%2Fmaster%2Fcsse_covid_19_data%2Fcsse_covid_19_time_series%2Ftime_series_covid19_confirmed_global.csv', parse_dates = [2], infer_datetime_format = True, usecols = c_usecols,skiprows=[1] ) #first row has duplicate headers that result in mistyped columns
confirmed['Date'] = pd.to_datetime(confirmed['Date'], errors='coerce')
r_usecols = ["Province/State", "Country/Region", "Date","Value", "ISO 3166-1 Alpha 3-Codes"]
recovered = pd.read_csv('https://data.humdata.org/hxlproxy/data/download/time_series_covid19_recovered_global_narrow.csv?dest=data_edit&filter01=merge&merge-url01=https%3A%2F%2Fdocs.google.com%2Fspreadsheets%2Fd%2Fe%2F2PACX-1vTglKQRXpkKSErDiWG6ycqEth32MY0reMuVGhaslImLjfuLU0EUgyyu2e-3vKDArjqGX7dXEBV8FJ4f%2Fpub%3Fgid%3D1326629740%26single%3Dtrue%26output%3Dcsv&merge-keys01=%23country%2Bname&merge-tags01=%23country%2Bcode%2C%23region%2Bmain%2Bcode%2C%23region%2Bsub%2Bcode%2C%23region%2Bintermediate%2Bcode&filter02=merge&merge-url02=https%3A%2F%2Fdocs.google.com%2Fspreadsheets%2Fd%2Fe%2F2PACX-1vTglKQRXpkKSErDiWG6ycqEth32MY0reMuVGhaslImLjfuLU0EUgyyu2e-3vKDArjqGX7dXEBV8FJ4f%2Fpub%3Fgid%3D398158223%26single%3Dtrue%26output%3Dcsv&merge-keys02=%23adm1%2Bname&merge-tags02=%23country%2Bcode%2C%23region%2Bmain%2Bcode%2C%23region%2Bsub%2Bcode%2C%23region%2Bintermediate%2Bcode&merge-replace02=on&merge-overwrite02=on&filter03=explode&explode-header-att03=date&explode-value-att03=value&filter04=rename&rename-oldtag04=%23affected%2Bdate&rename-newtag04=%23date&rename-header04=Date&filter05=rename&rename-oldtag05=%23affected%2Bvalue&rename-newtag05=%23affected%2Binfected%2Bvalue%2Bnum&rename-header05=Value&filter06=clean&clean-date-tags06=%23date&filter07=sort&sort-tags07=%23date&sort-reverse07=on&filter08=sort&sort-tags08=%23country%2Bname%2C%23adm1%2Bname&tagger-match-all=on&tagger-default-tag=%23affected%2Blabel&tagger-01-header=province%2Fstate&tagger-01-tag=%23adm1%2Bname&tagger-02-header=country%2Fregion&tagger-02-tag=%23country%2Bname&tagger-03-header=lat&tagger-03-tag=%23geo%2Blat&tagger-04-header=long&tagger-04-tag=%23geo%2Blon&header-row=1&url=https%3A%2F%2Fraw.githubusercontent.com%2FCSSEGISandData%2FCOVID-19%2Fmaster%2Fcsse_covid_19_data%2Fcsse_covid_19_time_series%2Ftime_series_covid19_recovered_global.csv', parse_dates = [2], infer_datetime_format = True, usecols = r_usecols, skiprows=[1])
d_usecols = ["Province/State", "Country/Region", "Date","Value", "ISO 3166-1 Alpha 3-Codes"]
deaths = pd.read_csv('https://data.humdata.org/hxlproxy/data/download/time_series_covid19_deaths_global_narrow.csv?dest=data_edit&filter01=merge&merge-url01=https%3A%2F%2Fdocs.google.com%2Fspreadsheets%2Fd%2Fe%2F2PACX-1vTglKQRXpkKSErDiWG6ycqEth32MY0reMuVGhaslImLjfuLU0EUgyyu2e-3vKDArjqGX7dXEBV8FJ4f%2Fpub%3Fgid%3D1326629740%26single%3Dtrue%26output%3Dcsv&merge-keys01=%23country%2Bname&merge-tags01=%23country%2Bcode%2C%23region%2Bmain%2Bcode%2C%23region%2Bsub%2Bcode%2C%23region%2Bintermediate%2Bcode&filter02=merge&merge-url02=https%3A%2F%2Fdocs.google.com%2Fspreadsheets%2Fd%2Fe%2F2PACX-1vTglKQRXpkKSErDiWG6ycqEth32MY0reMuVGhaslImLjfuLU0EUgyyu2e-3vKDArjqGX7dXEBV8FJ4f%2Fpub%3Fgid%3D398158223%26single%3Dtrue%26output%3Dcsv&merge-keys02=%23adm1%2Bname&merge-tags02=%23country%2Bcode%2C%23region%2Bmain%2Bcode%2C%23region%2Bsub%2Bcode%2C%23region%2Bintermediate%2Bcode&merge-replace02=on&merge-overwrite02=on&filter03=explode&explode-header-att03=date&explode-value-att03=value&filter04=rename&rename-oldtag04=%23affected%2Bdate&rename-newtag04=%23date&rename-header04=Date&filter05=rename&rename-oldtag05=%23affected%2Bvalue&rename-newtag05=%23affected%2Binfected%2Bvalue%2Bnum&rename-header05=Value&filter06=clean&clean-date-tags06=%23date&filter07=sort&sort-tags07=%23date&sort-reverse07=on&filter08=sort&sort-tags08=%23country%2Bname%2C%23adm1%2Bname&tagger-match-all=on&tagger-default-tag=%23affected%2Blabel&tagger-01-header=province%2Fstate&tagger-01-tag=%23adm1%2Bname&tagger-02-header=country%2Fregion&tagger-02-tag=%23country%2Bname&tagger-03-header=lat&tagger-03-tag=%23geo%2Blat&tagger-04-header=long&tagger-04-tag=%23geo%2Blon&header-row=1&url=https%3A%2F%2Fraw.githubusercontent.com%2FCSSEGISandData%2FCOVID-19%2Fmaster%2Fcsse_covid_19_data%2Fcsse_covid_19_time_series%2Ftime_series_covid19_deaths_global.csv', parse_dates = [2], infer_datetime_format = True, usecols = d_usecols, skiprows=[1])
deaths.head().T
# Dedupe tables and rename data column for merge
confirmed = confirmed.drop_duplicates(keep = 'last') #many duplicat date from multiple reports of data on same day at different times within day
recovered = recovered.drop_duplicates(keep = 'last') #many duplicat date from multiple reports of data on same day at different times within day
deaths  = deaths.drop_duplicates(keep = 'last') #many duplicat date from multiple reports of data on same day at different times within day
confirmed.rename(columns = {'Value':'confirmed'}, inplace = True) 
recovered.rename(columns = {'Value':'recovered'}, inplace = True) 
deaths.rename(columns={'Value': 'deaths'}, inplace=True)
confirmed.head(3).T
#%% Population data from World Bank
# https://api.worldbank.org/indicator/sp.pop.totl?format=csv
# https://data.world/worldbank/total-population-per-country
population = pd.read_csv('https://api.worldbank.org/indicator/sp.pop.totl?format=csv',  usecols = ["Country Name","Country Code", "2018", "2019"] )#skiprows = 4,
population.info()
population.tail()
#%% Merge three table, Outer, drop duplicated Lat/Long columns
confirmed.info()
cov = pd.merge(confirmed,recovered, how = 'outer', on=['Province/State', 'Country/Region', 'Date'])
cov = pd.merge(cov, deaths, how = 'outer', on=['Province/State', 'Country/Region', 'Date'])
cov.rename(columns={'ISO 3166-1 Alpha 3-Codes': 'code'}, inplace=True)
cov = pd.merge(cov,population, how = 'left', left_on= 'code', right_on = 'Country Code')#.drop(unwanted,1) 
cov.drop(cov.index[0], inplace = True) #drop second level headers
cov['code'] = cov.apply(lambda x: x['ISO 3166-1 Alpha 3-Codes_x'] if pd.isnull(x['code']) else x['code'], axis = 1)# grab country code if missing in "code" column
cov['code'] = cov.apply(lambda x: x['ISO 3166-1 Alpha 3-Codes_y'] if pd.isnull(x['code']) else x['code'], axis = 1)# grab country code if missing in "code" column
cov.drop(['ISO 3166-1 Alpha 3-Codes_x', 'ISO 3166-1 Alpha 3-Codes_y', 'Country Code', "Country Name"], 1,inplace = True) #drop duplicate columns
cov.head().T
cov.info()
cov['Last_Update'] = pd.to_datetime(cov['Date'], errors = 'coerce').dt.date
cov['Last_Update'] = pd.to_datetime(cov['Last_Update'], errors = 'coerce')
cov[['Lat','Long']] = cov[['Lat','Long']].astype(float)
cov.head().T
cov[['confirmed','recovered','deaths']] = cov[['confirmed','recovered','deaths']].fillna(method='ffill')
cov['log_conf'] = np.log(cov['confirmed'])
#%% Within a Country,SUM all State values by Date
np.seterr(divide = 'ignore') #turn off warning for dividing by missing values
cov2 = cov.groupby(['Country/Region','Last_Update'], as_index = False).agg({'confirmed':'sum','deaths':'sum','recovered':'sum', '2018':'max', 'Lat':'max', 'Long':'max'})
#cov22 = cov.groupby(['Country_Region','Last_Update'], as_index = False).agg({'Confirmed':'sum','Deaths':'sum','Recovered':'sum', 'pop2020':'max'})
cov2.rename(columns={'2018': 'pop'}, inplace=True)
cov2.sort_values(['Country/Region','Last_Update'], ascending= [True,True], inplace=True)
cov2.set_index(['Country/Region', 'Last_Update'], inplace = True)
cov2['3_rollmean_confirmed'] = cov2.groupby(level=0)['confirmed'].rolling(window=3).mean().values
cov2['5_rollmean_confirmed'] = cov2.groupby(level=0)['confirmed'].rolling(window=5).mean().values
cov2['3shift3_Confirmed'] = cov2.groupby(level=0)['confirmed'].shift(3).rolling(window=3).mean()
cov2['5shift5_Confirmed'] = cov2.groupby(level=0)['confirmed'].shift(5).rolling(window=5).mean()
cov2['Rolling_Rank'] = cov2['5_rollmean_confirmed']/cov2['5shift5_Confirmed']
cov2['log_confirmed'] = np.log(cov2['confirmed'])
cov2['cum_Conf'] =  cov2.groupby(level=0)['confirmed'].cumsum()
cov2['cum_Recv'] = cov2.groupby(level=0)['recovered'].cumsum()
cov2['recv_rate'] = cov2['cum_Recv']/cov2['cum_Conf']
cov2['pct_conf'] = cov2['cum_Conf']/cov2['pop']
cov2['pct_recv'] = cov2['cum_Recv']/cov2['pop']
cov2['pct_conf'] = cov2['cum_Conf']/cov2['pop']
#cov2[['ConfChg','RecvChg']] = cov2.sort_values(['Country_Region','Last_Update'], ascending= [True,True]).groupby(level=0)[['cum_Conf','cum_Recv' ]].pct_change()
cov2[['ConfChg','RecvChg']] = cov2.groupby(level=0)[['cum_Conf','cum_Recv' ]].pct_change()
world_pop = cov2['pop'].sum() # missing data from country name mismatches at merge
conf_rt = cov2['Rolling_Rank'].mean()
conf_rt
cov2.tail(14)
#cov2.to_csv(r'C:\EliPersonal\Python\Datasets\Doit\covid_agg.csv', encoding = 'utf-8') # save data set with features to csv
#%% Ordered by day (NOT by country/day)
cov_all =  cov2.reset_index(level=0, drop=True)
cov_all.sort_values('Last_Update', ascending= True, inplace=True)
cov_all['worldwide_cum_confirmed'] = cov_all['confirmed'].cumsum()
print ('Total Confirmed Covid Cases Worldwide: ',cov_all['confirmed'].cumsum().max())
#%% Save Master data to csv for PowerBI, and RANKINGS
agg_data = cov2.reset_index()    
agg_data.sort_values(['Country/Region','Last_Update'], ascending= [True,True], inplace=True)
agg_data.head()
Rankings = agg_data.loc[:,['Country/Region','Last_Update', 'Rolling_Rank', 'Lat','Long']]
Rankings = agg_data.groupby('Country/Region')['Rolling_Rank'].last()
Rankings = Rankings.reset_index()
Rankings.dropna().sort_values(by = ['Rolling_Rank'], ascending = True)
agg_data.to_csv(r'C:\EliPersonal\Python\Datasets\Doit\agg_data.csv', encoding = 'utf-8', index = False) # save data set with features to csv
#agg_data.query('(Country_Region == "Israel") and ()').head() #query doesn't work with slashes in column names
#test Azure created csv
# zz = pd.read_csv('https://covid19worldwideactivity-elilieberman.notebooks.azure.com/j/lab/tree/agg_data.csv', lineterminator='\n', encoding = 'utf-8', engine='python', sep=',', error_bad_lines=False)
#%% Plot Country Trend
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
country_selected = 'Israel'  #used for graphs and ARIMA

      
plot_data = cov2.reset_index()    
plot_data.sort_values(['Country/Region','Last_Update'], ascending= [True,True], inplace=True)
plot_data.head()
plot_data = plot_data.loc[:,['Country/Region','Last_Update','log_confirmed', '5_rollmean_confirmed','pct_conf', 'pct_recv']]
plots = plot_data[plot_data['Country/Region'] == country_selected]

plot_title_a = 'COVID Trend using Rolling Mean, ' + country_selected 
sns.set(style="white", rc={"lines.linewidth": 3})
fig, ax1 = plt.subplots(figsize=(8,4))
sns.lineplot(x='Last_Update', 
             y='5_rollmean_confirmed',
             data = plots,
             color='r',
             marker='<',
             linewidth = 1,
             linestyle = '-',
             ax=ax1)
ax1.tick_params('both', labelrotation=45)
ax1.xaxis.grid(True)
ax1.yaxis.grid(True)
ax1.set_xlabel('')
plt.title(plot_title_a)
plt.savefig(r'C:\EliPersonal\Python\Datasets\Doit\COVID_Trend_Israel.png')
sns.set()
#%% Plot data as a Percent of POPULATION
plot_title_b = 'COVID Confirmed v. Recovering, Population Data ' + country_selected +'\n(Confirmed is red line, Recovered is green line)'
sns.set(style="white", rc={"lines.linewidth": 3})
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1,figsize=(6,3), sharex = True)
sns.lineplot(x='Last_Update',
            y='pct_conf', 
            data = plots,
            linewidth = 1,
            color='r',
            marker='o',
            linestyle = ':',
            ax=ax1)
sns.lineplot(x='Last_Update', 
             y='pct_recv',
             data = plots,
             color='g',
             marker='D',
             linewidth = 1,
             linestyle = '-',
             ax=ax2)
ax2.tick_params('both', labelrotation=45)
ax1.xaxis.grid(True)
ax2.xaxis.grid(True)
ax1.yaxis.grid(True)
ax2.yaxis.grid(True)
ax1.set_xlabel('')
ax1.set_title(plot_title_b, pad =20)
plt.show()
plt.savefig(r'C:\EliPersonal\Python\Datasets\Doit\COVID_Israel_Population_Data.png')
sns.set()
#%% Read Israel SYMPTOM data using JSON api, convert RESULTS dictionary to dataframe
import pandas as pd
isr_sym = pd.read_json('https://data.gov.il/api/action/datastore_search?resource_id=d337959a-020a-4ed3-84f7-fca182292308&limit=300000')
isr_sym = pd.DataFrame.from_dict(isr_sym['result']['records']).reset_index(drop = True)
isr_sym = isr_sym.drop_duplicates(keep = 'last') #clean away any duplicate lines of data
isr_sym.replace(to_replace=['חיובי', 'שלילי'], value=[0, 1], inplace = True) #fix categorical values
isr_sym.replace(to_replace=['אחר','נקבה', 'זכר'], value=[0, 1,1], inplace = True) #fix categorical values
isr_sym['test_date'] = pd.to_datetime(isr_sym['test_date'], errors = 'coerce').dt.date
isr_sym['male'] = isr_sym.apply(lambda x: 1 if x.gender == "M" else 0, axis = 1) #convert string values to numeric
isr_sym['over60'] = isr_sym['age_60_and_above'].eq('Yes').astype(int) #convert string values to numeric
isr_sym[['age_60_and_above','over60']].tail(20)
isr_sym.info()

#%%  Linear Regression Model, Israel Test Data
labels = ['cough', 'fever', 'sore_throat','shortness_of_breath', 'head_ache', 'male', 'over60'] #'test_date', 'test_indication',
X = isr_sym[labels]
X = X.apply(pd.to_numeric, errors='coerce')
y = isr_sym.corona_result
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
imp = SimpleImputer(strategy='mean')
X = imp.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=1)

# import, instantiate model, fit
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X_train, y_train)

# print the intercept and coefficients
print('Intercept: ',linreg.intercept_)
print('Coefficients/Weights: ', linreg.coef_)
isr_sym_ratio = pd.DataFrame(list(zip(labels, abs(linreg.coef_))),columns=['symptom', 'weight'])
print(isr_sym_ratio.sort_values(by = 'weight', ascending = False))
#plt.close
chart_info = 'Top Symptoms, COVID Coefficient Weights, Source: gov.co.il\n' + 'Number Test Conducted ' + str(max(isr_sym._id))+ ',' + ' Data Updated ' + str(max(isr_sym.test_date))
isr_sym_ratio.sort_values(by = 'weight', ascending = False).head(5).plot(kind='bar', x = 'symptom', y = 'weight', title = chart_info, color=['r', 'g', 'b', 'y', 'm', 'k', 'c'], legend = None)
plt.ylabel('Weight/Importance', fontsize=10)
plt.xticks(rotation=45, ha="right")   #fpr x-axis legend labels to fit properly
plt.tight_layout() #fpr x-axis legend labels to fit properly
plt.savefig(r'C:\EliPersonal\Python\Datasets\Doit\COVID_Israel_Testing_Data.png')
#%% ARIMA TIME SERIES ANALYSIS (TSA) Forecasting
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
from math import sqrt

#aggregated dailyvalues, of a SINGLE country, by day
arima_df = agg_data[agg_data['Country/Region'] == country_selected].loc[:,['Last_Update', 'confirmed']] 
# Resampling of daily reporting
arima_df.set_index('Last_Update', inplace = True)
arima_df.tail().T
arima_df.info()
arima_df = arima_df.resample('1D').apply(np.sum)

#Checking trend and autocorrelation
def initial_plots(time_series, num_lag):
    #Original timeseries plot
    plt.figure(1)
    plt.plot(time_series)
    plt.title('Original data across time')
    plt.figure(2)
    plot_acf(time_series, lags = num_lag)
    plt.title('Autocorrelation plot')
    plot_pacf(time_series, lags = num_lag)
    plt.title('Partial autocorrelation plot')
    plt.show()
    
#Augmented Dickey-Fuller test for stationarity
#checking p-value
# adfuller(arima_df.confirmed.values)  #all values, p-value, lag, number obs, 
print('p-value: {}'.format(adfuller(arima_df)[1]))
print('Lag-value: {}'.format(adfuller(arima_df)[2]))
print('Number of Observations: {}'.format(adfuller(arima_df)[3]))
lag = adfuller(arima_df)[2]

#storing differenced series
diff_series = arima_df.diff(periods=1)
# fit model using Lag 4 from previous analysis, Diff 1 because daily values, Rolling window 3 given the prevailing window for this data
model = ARIMA(arima_df, order=(lag,1,3)) #lag-1 versus lag, prevents SVD error
model_fit = model.fit(disp=0)
print(model_fit.summary())
# Forecast for the next 10 days, beginning 45 days back
#plt.close()
plot_title_arima_pred = 'ARIMA Predicted, Confirmed Cases, ' + country_selected +'\n(looking back 45 days, looking forward 10 days, last confirmed case:' + str(max(isr_sym.test_date)) +')'
forecast = model_fit.predict(start = len(arima_df),end = (len(arima_df)-1) + 10, typ = 'levels')
# Plot the forecast values 
sns.set(style="white", rc={"lines.linewidth": 3})
fig, ax1 = plt.subplots(figsize=(8,4))
sns.lineplot(x='Last_Update',
            y= 'confirmed', 
            data = arima_df.reset_index()[-45:],
            label = 'recorded', 
            linewidth = 2,
            color='r',
            linestyle = ':',
            ax=ax1)
sns.lineplot(x='index', 
             y=0,
             data = forecast.reset_index(),
             label = 'predicted', 
             color='m', 
             marker = 'x',
             linewidth = 3,
             linestyle = '-',
             ax=ax1)
ax1.tick_params('both', labelrotation=45)
ax1.xaxis.grid(True)
ax1.yaxis.grid(True)
ax1.set_xlabel('')
ax1.set_ylabel('Confirmed Cases and Predicted')
plt.title(plot_title_arima_pred)
plt.show()
plt.savefig(r'C:\EliPersonal\Python\Datasets\Doit\COVID_Israel_Forecast_Data.png')
sns.set()





