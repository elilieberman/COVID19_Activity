import numpy as np
import pandas as pd
import os
import sys # for utilities that check memory issues
pd.set_option('display.max_columns', None)
pd.option_context('display.max_colwidth', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None) #dont truncate column value display
pd.set_option('display.max_columns', None) # .head will show all
pd.set_option('display.max_rows', None) # .head will show all
#%% https://worldpopulationreview.com/#liveWorldPop population stats
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
cov2[['ConfChg','RecvChg']] = cov2.groupby(level=0)[['cum_Conf','cum_Recv' ]].pct_change()
world_pop = cov2['pop'].sum() # missing data from country name mismatches at merge
conf_rt = cov2['Rolling_Rank'].mean()
conf_rt
cov2.tail(14)
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
#%% Plot Country Trend
import seaborn as sns
import matplotlib.pyplot as plt

country_selected = 'Italy'  #used for graphs and ARIMA

plot_title_a = 'COVID Trend, using Log and Rolling Mean' + country_selected
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
ax1.xaxis.grid(True)
ax1.yaxis.grid(True)
ax1.set_xlabel('')
plt.title(plot_title_a)
plt.show()
plt.savefig(r'C:\EliPersonal\Python\Datasets\Doit\COVID_Trend_Italy.png')
sns.set()
#%% Plot POPULATION Relative Changes
plt.close()
plot_title_b = 'COVID Confirmed v. Recovering, Population Data ' + country_selected
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
ax1.xaxis.grid(True)
ax1.yaxis.grid(True)
ax1.set_xlabel('')
plt.title(plot_title_b)
plt.show()
plt.savefig(r'C:\EliPersonal\Python\Datasets\Doit\COVID_PerCapita_Activity_Italy.png')
sns.set()