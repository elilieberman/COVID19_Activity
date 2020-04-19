# COVID19_Activity
Python Script with visuals for live Tracking COVID19 confirmed, recovered and mortality stats
Using the URL's from the three HUMDATA.org files, complete Confirmed, Recvered and Death stats are provided after merging the live data.
The HUMDATA is deduped and aggregated "per day" for proper time series frequency/spacing.
The WORLDBANK URL provides country population data, to give context to the raw COVID case data, i.e. what the numbers mean as a percentage of the respective country population
The 5 day rolling mean, is provided for the Country Trend visual (dual axis of actual v. rolling mean)
To modify the graphs for another Country update the variable "country_selected" (line 96)
The agg_data csv is the sourcefile for related PowerBI dashboard, check it out...
https://app.powerbi.com/groups/me/dashboards/4414473e-b527-439d-b735-28e9d365058c?ctid=43eaa772-f7b9-49c0-a310-6db2e0414f42
