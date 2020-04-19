# COVID19_Activity
1. Python Python 3.7.6 Script, with visuals for live Tracking COVID19 confirmed, recovered and mortality stats.  <br /> 
2. Using the URL's from the three HUMDATA.org files, updated CONFIRMED, RECOVERED and DEATH case data are merged into a daily activity report. <br /> 
3. The HUMDATA is deduped and aggregated "per day" for proper time series frequency/spacing. <br /> 
4. The WORLDBANK URL provides country population data, to give context to the raw COVID case data, i.e. what the numbers mean as a percentage of the respective country population. <br /> 
5. The 5 day rolling mean, is provided for the Country Trend visual (dual axis of actual v. rolling mean) <br /> 
6. To modify the graphs for another Country update the variable "country_selected" (line 96) <br /> 
7. The agg_data csv is the sourcefile for related PowerBI dashboard, check it out...
https://app.powerbi.com/groups/me/dashboards/4414473e-b527-439d-b735-28e9d365058c?ctid=43eaa772-f7b9-49c0-a310-6db2e0414f42
