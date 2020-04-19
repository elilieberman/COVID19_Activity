# COVID19_Activity
1. Python Python 3.7.6, IPython 7.13.0, Script, for Jupyter lines 5-10 (display environment variables) may need to be commented.<br /> 
2. COVID19 dynamic reporting and visuals,  of CONFIRMED, RECOVERED and MORTALITY stats.<br /> 
3. The three URL's from HUMDATA.org provide updated CONFIRMED, RECOVERED and DEATH case data, which are merged into a daily activity report.<br /> 
4. The HUMDATA is then deduped and aggregated "per day" for proper time series frequency/spacing.<br /> 
5. The WORLDBANK URL provides country population data, to give context to the raw COVID case data, i.e. what the numbers mean as a percentage of the respective country population, used in the "Covid Confirmed v. Recovering, Population Data" visual.<br /> 
6. The 5 day rolling mean, is provided for the "COVID Activity and Trend" visual (dual axis of actual v. rolling mean).<br /> 
7. To modify the graphs for another Country update the variable "country_selected" (line 96) <br /> 
8. The agg_data csv is the sourcefile for related PowerBI dashboard, check it out...
https://app.powerbi.com/groups/me/dashboards/4414473e-b527-439d-b735-28e9d365058c?ctid=43eaa772-f7b9-49c0-a310-6db2e0414f42
