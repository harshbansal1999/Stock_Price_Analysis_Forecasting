import pandas as pd
from Data_Fetch import data_fetch
from EDA import current_price,today_price,overall,returns_today,overall_return,all_var,volume_eda
from Index_Ranking import sensex,nifty
from FM_Backtesting import backtesting
from FM_Bollinger import bollinger
from FM_CAPM import capm
from FM_GARCH import garch
from FM_HVnSR import rv_sr
from FM_Monte_Carlo import monte_carlo_simulation
from Portfolio import portfolio_report
from FM_RSI import rsi
from FM_Seasonal_Decomposition import seasonal_decompose
from FM_Stylized_Facts import stylized_facts
from FM_Time_Series_Model import time_series_model
from Forecasting import forecast
from News import news
#from News import news


#5
#Data Fetch
data,df,name,tick = data_fetch()
print('\n\n')

ctr=0

while(ctr!=1):

    print("1.\tForecasting\n2.\tNews with Sentiment Analysis\n3.\tEDA(Current Price)\n4.\tEDA(Today Price)\n5.\tEDA(Last 6 months)\n6.\tEDA(Current Returns)\n7.\tEDA(Overall Returns)\n8.\tEDA(OHLC)\n9.\tEDA(Volume)\n10.\tIndex Ranking\n11.\tBacktesting\n12.\tBollinger Model\n13.\tCAPM\n14.\tGARCH\n15.\tHistorical Volatitlity And Sharp Ratio\n16.\tMonte Carlo Simulation\n17.\tRSI\n18.\tSeasonal Decompose\n19.\tStylized Facts\n20.\tTime Series\n21.\tPortfolio")
    print('\nEnter -1 to exit')

    inp=int(input("Enter the input:"))

    if(inp==1):
        forecast(tick,name)
    elif(inp==2):
        news(tick)
    elif(inp==3):
        current_price(df,data,"Close",name)
    elif(inp==4):
        today_price(df,data,"Close",name)
    elif(inp==5):
        overall(data,"Close",name)
    elif(inp==6):
        returns_today(df,data,name)
    elif(inp==7):
        overall_return(data,name)
    elif(inp==8):
        all_var(data,name)
    elif(inp==9):
        volume_eda(data,name)
    elif(inp==10):
        nifty()
        sensex()
    elif(inp==11):
        backtesting(tick,name)
    elif(inp==12):
        bollinger(tick,name)
    elif(inp==13):
        capm(tick,name)
    elif(inp==14):
        garch(tick,name)
    elif(inp==15):
        rv_sr(tick,name)
    elif(inp==16):
        monte_carlo_simulation(tick,name)
    elif(inp==17):
        rsi(tick,name)
    elif(inp==18):
        seasonal_decompose(tick,name)
    elif(inp==19):
        stylized_facts(tick,name)
    elif(inp==20):
        time_series_model(tick,name)
    elif(inp==21):
        portfolio_report()
    elif(inp==-1):
        break

#Forecasting

'''
forecast(tick,name)
'''

#News
'''
news(tick)
'''

#EDA
#1. Current Price

'''
current_price(df,data,"Close",name)
current_price(df,data,"Open",name)
current_price(df,data,"High",name)
current_price(df,data,"Low",name)
'''

#2. Today's Price
'''
today_price(df,data,"Close",name)
today_price(df,data,"Open",name)
today_price(df,data,"High",name)
today_price(df,data,"Low",name)
'''

#3. Overall Price (Last 6 months)
'''
overall(data,"Close",name)
overall(data,"Open",name)
overall(data,"High",name)
overall(data,"Low",name)
'''

#4. Returns Current
'''
returns_today(df,data,name)
'''

#5. Returns Overall
'''
overall_return(data,name)
'''

#6. All Var
'''
all_var(data,name)
'''
#7. Volume
'''
volume_eda(data,name)
'''

#Index Ranking
'''
nifty()
sensex()
'''

#Financial Modelling

#1. Backtesting Strategy
'''
backtesting(tick,name)
'''

#2. Bollinger Band
'''
bollinger(tick,name)
'''

#3.CAPM
'''
capm(tick,name)
'''

#4. GARCH
'''
garch(tick,name)
'''

#5. Historical VOlatitlity And Sharp Ratio
'''
rv_sr(tick,name)
'''

#6. Monte Carlo Simulation
'''
monte_carlo_simulation(tick,name)
'''

#7. RSI
'''
rsi(tick,name)
'''

#8. Seasonal Decompose
'''
seasonal_decompose(tick,name)
'''

#9. Stylized Facts
'''
stylized_facts(tick,name)
'''

#10. Time Series Model
'''
time_series_model(tick,name)
'''

#Portfolio
'''
portfolio_report()
'''

#Reinforcement
'''
reinforcement(tick,name)
'''
