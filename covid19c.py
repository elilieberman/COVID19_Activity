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
country_selected = 'Italy'  #used for graphs and ARIMA
plot_title_a = 'COVID Activity, ' + country_selected
      
plot_data = cov2.reset_index()    
plot_data.sort_values(['Country/Region','Last_Update'], ascending= [True,True], inplace=True)
plot_data.head()
plot_data = plot_data.loc[:,['Country/Region','Last_Update','log_confirmed', '5_rollmean_confirmed','pct_conf', 'pct_recv']]
plots = plot_data[plot_data['Country/Region'] == country_selected]

plt.close()
sns.set(style="white", rc={"lines.linewidth": 3})
fig, ax1 = plt.subplots(figsize=(8,4))
ax2 = ax1.twinx()
sns.lineplot(x='Last_Update',
            y='log_confirmed', 
            data = plots,
            linewidth = 1,
            color='b',
            marker='o',
            linestyle = ':',
            ax=ax1)
sns.lineplot(x='Last_Update', 
             y='5_rollmean_confirmed',
             data = plots,
             color='g',
             marker='D',
             linewidth = 1,
             linestyle = '-',
             ax=ax2)
ax1.tick_params('both', labelrotation=90)
#ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
#ax2.yaxis.set_major_locator(ticker.MultipleLocator(5))
ax1.xaxis.grid(True)
ax1.yaxis.grid(True)
ax1.set_xlabel('')
plt.title(plot_title_a)
plt.show()
plt.savefig(r'C:\EliPersonal\Python\Datasets\Doit\COVID_Trend_Italy.png')
sns.set()
#%% Plot POPULATION Relative Changes
plt.close()
plot_title_b = 'Covid Confirmed v. Recovering, ' + country_selected
sns.set(style="white", rc={"lines.linewidth": 3})
fig, ax1 = plt.subplots(figsize=(8,4))
ax2 = ax1.twinx()
sns.lineplot(x='Last_Update',
            y='pct_conf', 
            data = plots,
            linewidth = 1,
            color='b',
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
ax1.tick_params('both', labelrotation=90)
#ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
#ax2.yaxis.set_major_locator(ticker.MultipleLocator(5))
ax1.xaxis.grid(True)
ax1.yaxis.grid(True)
ax1.set_xlabel('')
plt.title(plot_title_b)
plt.show()
plt.savefig(r'C:\EliPersonal\Python\Datasets\Doit\COVID_PerCapita_Activity_Italy.png')
sns.set()
#%% Read Israel Symptom data using JSON api, convert RESULTS dictionary to dataframe
import pandas as pd
#isr5_sym = pd.read_json('https://data.gov.il/api/action/datastore_search?resource_id=d337959a-020a-4ed3-84f7-fca182292308&limit=5')
#isr5_sym.head().T
isr_sym = pd.read_json('https://data.gov.il/api/action/datastore_search?resource_id=d337959a-020a-4ed3-84f7-fca182292308&limit=300000')
isr_sym = pd.DataFrame.from_dict(isr_sym['result']['records']).reset_index(drop = True)
isr_sym = isr_sym.drop_duplicates(keep = 'last') #clean away any duplicate lines of data
isr_sym.replace(to_replace=['חיובי', 'שלילי'], value=[0, 1], inplace = True) #fix categorical values
isr_sym.replace(to_replace=['אחר','נקבה', 'זכר'], value=[0, 1,1], inplace = True) #fix categorical values
isr_sym['test_date'] = pd.to_datetime(isr_sym['test_date'], errors = 'coerce').dt.date
isr_sym['male'] = isr_sym.apply(lambda x: 1 if x.gender == "M" else 0, axis = 1)
isr_sym['age60'] = isr_sym['age_60_and_above'].eq('Yes').astype(int)
#isr_sym['test_date'] = isr_sym['test_date'].astype(str).astype('datetime64[D]')
#isr_sym['test_date'] = isr_sym['test_date'].dt.date
#isr_sym.to_numeric([['cough','fever', 'sore_throat', inplace= True, errors)
#isr_sym.iloc[:,2:] = isr_sym.iloc[:,2:].astype(str).astype(int, errors = 'ignore')
isr_sym[['age_60_and_above','age60']].tail(20)
isr_sym.info()

#%%  Linear Regression Model
#n = isr_sym.isin(['Null']).any()
labels = ['cough', 'fever', 'sore_throat','shortness_of_breath', 'head_ache', 'male', 'age60'] #'test_date', 'test_indication',
X = isr_sym[labels]
X = X.apply(pd.to_numeric, errors='coerce')
y = isr_sym.corona_result
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
imp = SimpleImputer(strategy='mean')
X = imp.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=1)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# import, instantiate model, fit
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X_train, y_train)

# print the intercept and coefficients
print(linreg.intercept_)
print(linreg.coef_)
list(zip(labels, linreg.coef_))
isr_sym_ratio = pd.DataFrame(list(zip(labels, abs(linreg.coef_))),columns=['symptom', 'weight'])
isr_sym_ratio.sort_values(by = 'weight', ascending = False)
plt.close
isr_sym_ratio.head(5).plot(kind='bar', x = 'symptom', y = 'weight', title = 'COVID Symptom Predictors', color=['r', 'g', 'b', 'y', 'm', 'k', 'c'], label = '')
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
#%% LOGISTIC MODEL
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(solver = 'lbfgs')
logreg.fit(X, y)

# predict the response for new observations
logreg.predict(X_new)

from sklearn import metrics
y_pred = logreg.predict(X)
len(y_pred)
print(metrics.accuracy_score(y, y_pred))
#print(logreg.predict_proba([[]])

#%% TIME SERIES ANALYSIS (TSA)
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
from math import sqrt

#aggregated dailyvalues, of a SINGLE country, by day
arima_df = agg_data[agg_data['Country/Region'] == country_selected].loc[:,['Last_Update', 'confirmed']] 
# Resampling of daily reporting
# arima_df.index = pd.DatetimeIndex(arima_df.index)
#arima_df.sort_values('Date', ascending= True, inplace=True) #otherwise throws "monotonic" date Value index errors
arima_df.set_index('Last_Update', inplace = True)
arima_df.head().T
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
# adfuller(arima_df.confirmed.values)  #all values,test stst, , p-value, lag, number obs, 
print('p-value: {}'.format(adfuller(arima_df)[1]))
print('Lag-value: {}'.format(adfuller(arima_df)[2]))
print('Number of Observations: {}'.format(adfuller(arima_df)[3]))

#plotting
initial_plots(arima_df, 12)
plot_acf(arima_df.values, zero=True)
plt.plot()


#storing differenced series
diff_series = arima_df.diff(periods=1)

#Augmented Dickey-Fuller test for stationarity
#checking p-value
print('p-value: {}'.format(adfuller(diff_series.dropna())[1]))
initial_plots(diff_series.dropna(), 30)

# fit model using Lag 12, Diff 1, Rolling window 3
model = ARIMA(arima_df, order=(12,1,3))
model_fit = model.fit(disp=0)
print(model_fit.summary())
# plot residual errors
residuals = pd.DataFrame(model_fit.resid)
residuals.plot()
plt.title('ARIMA Residual (Line)')
plt.show()
residuals.plot(kind='kde')
plt.title('ARIMA Residual (KDE)')
plt.show()
print(residuals.describe())


# Run ARIMA Model
plot_title_arima_perf = 'ARIMA Performance, ' + country_selected
X = arima_df.values
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
	model = ARIMA(history, order=(0,1,0))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
# plot
plt.plot(test)
plt.plot(predictions, color='red')
plt.title(plot_title_arima_perf)
plt.show()

# Forecast for the next 10 days
plot_title_arima_pred = 'ARIMA Predicted, Confirmed Cases for, ' + country_selected
forecast = model_fit.predict(start = len(arima_df),end = (len(arima_df)-1) + 14, typ = 'levels')
# Plot the forecast values 
#arima_df.isin([-36876.35]).any()
arima_df['confirmed'].plot(figsize = (12, 8), legend = True) 
plt.title(plot_title_arima_pred)
forecast.plot(legend = True) 
plt.show()

#%%
'''isr_tests = pd.read_excel(r'C:\EliPersonal\Python\Datasets\corona_lab_tests_ver003.xlsx', nrows = 5, parse_dates = True) #take sample first 5 records
isr_tests = pd.read_excel(r'C:\EliPersonal\Python\Datasets\corona_lab_tests_ver003.xlsx', parse_dates = True)
isr_tests.replace(to_replace=['חיובי', 'שלילי'], value=[0, 1], inplace = True) #fix categorical values
isr_tests.head().T
isr_tests.info()'''

#%%
'''
isr5_tests = pd.read_json('https://data.gov.il/api/action/datastore_search?resource_id=dcf999c1-d394-4b57-a5e0-9d014a62e046&limit=10')
isr5_tests = pd.DataFrame.from_dict(isr5_tests['result']['records']).reset_index(drop = True)
isr5_tests[['test_date','result_date']] = isr5_tests[['test_date', 'result_date']].astype('datetime64[D]')
isr5_tests.head()
isr5_tests.info()
'''
isr_tests = pd.read_json('https://data.gov.il/api/action/datastore_search?resource_id=dcf999c1-d394-4b57-a5e0-9d014a62e046&limit=300000')
isr_tests = pd.DataFrame.from_dict(isr_tests['result']['records']).reset_index(drop = True)
isr_tests = isr_tests.drop_duplicates(keep = 'last') #clean away any duplicate lines of data
#isr_tests[['test_date', 'result_date']] = isr_tests[['test_date', 'result_date']].astype('datetime64[D]')
isr_tests['test_date'] = pd.to_datetime(isr_tests['test_date'], errors = 'coerce').dt.date
isr_tests['result_date'] = pd.to_datetime(isr_tests['result_date'], errors = 'coerce').dt.date
isr_tests.replace(to_replace=['חיובי', 'שלילי'], value=[0, 1], inplace = True) #fix categorical values
isr_tests.head().T
isr_tests.info()
len(isr_tests)
'''
isr_tests.sort_values(['_id','corona_result', 'result_date'], ascending= [True,True,True], inplace=True)
isr_tests.reset_index(inplace=True)
isr_tests.set_index(['_id', 'corona_result'], inplace = True)
cov2['5_rollmean_confirmed'] = cov2.groupby(level=0)['confirmed'].max().values
#isr_cov = pd.merge(isr_sym,isr_tests, how = 'left', on=['_id'])
'''
isr_recv = pd.read_json('https://data.gov.il/api/action/datastore_search?resource_id=8455d49f-ce32-4f8f-b1d4-1d764660cca3&limit=5')
isr_recv = pd.DataFrame.from_dict(isr_recv['result']['records']).reset_index(drop = True)
isr_recv = isr_recv.drop_duplicates(keep = 'last') #clean away any duplicate lines of data
isr_recv.replace(to_replace=['נקבה', 'זכר'], value=['M', 'F'], inplace = True) #fix categorical values
isr_recv.head().T
isr_recv.info()
len(recv_tests)
