# COVID19_Activity
1. Python Python 3.7.6, IPython 7.13.0, Script, for Jupyter lines 5-10 (display environment variables) may need to be commented.<br /> 
2. COVID19 dynamic reporting and visuals,  of CONFIRMED, RECOVERED and MORTALITY stats.<br /> 
3. The three URL's from HUMDATA.org provide updated CONFIRMED, RECOVERED and DEATH case data, which are merged into a daily activity report.<br /> 
4. The HUMDATA is then deduped and aggregated "per day" for proper time series frequency/spacing.<br /> 
5. The WORLDBANK URL provides country population data, to give context to the raw COVID case data, i.e. what the numbers mean as a percentage of the respective country population, used in the "Covid Confirmed v. Recovering, Population Data" visual.<br /> 
6. The 5 day rolling mean, is provided for the "COVID Activity and Trend" visual (dual axis of actual v. rolling mean).<br /> 
7. To modify the graphs for another Country update the variable "country_selected" (line 96) <br />
* The "COVID Trend" plot shows the Trend of confirmed cases, plots trendlines using the "log" and "rolling mean" (visible flattening of the curve) <br />
* The "COVID Confirmed and Recovery", displays these trends "a percentage of the population"<br />
https://app.powerbi.com/groups/me/dashboards/4414473e-b527-439d-b735-28e9d365058c?ctid=43eaa772-f7b9-49c0-a310-6db2e0414f42 <br /> 
8. The covid19_Israel has added an SYMPTOM chart and demonstative Prediction data (as a on-practioner), data source is data.gov.il<br />
* The "COVID Trend" plot shows the Trend of confirmed cases, plots trendlines using the "log" and "rolling mean" (visible flattening of the curve) <br />
* The "COVID Confirmed and Recovery", displays these trends "a percentage of the population"<br />
* The Bar plot, shows which symptoms are seem most indiciative of COVID<br />
* The final plot, is a simple Time Series forecasting model (ARIMA) for new cases in the next 7 days<br />


