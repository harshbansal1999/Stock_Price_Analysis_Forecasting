import pandas as pd
import yfinance as yf
import statsmodels.api as sm
import datetime

def capm(tick,name):

    RISKY_ASSET = tick
    MARKET_BENCHMARK = '^GSPC'
    end = datetime.datetime.now()
    start = datetime.datetime.now() - datetime.timedelta(days=60)

    df = yf.download([RISKY_ASSET, MARKET_BENCHMARK], start=start, end=end, auto_adjust=True, progress=False)

    X = df['Close'].rename(columns={RISKY_ASSET: 'asset', MARKET_BENCHMARK: 'market'}).resample('M').last().pct_change().dropna()
    covariance = X.cov().iloc[0,1]
    benchmark_variance = X.market.var()
    beta = covariance / benchmark_variance

    print("------------------------------------- Analysis Based on Last 2 months Data -------------------------------------")
    print("------------------------------------- {} -------------------------------------\n".format(name))
    if(beta<=-1):
        print('The asset moves in the opposite direction as the benchmark and in a greater amount than the negative of the benchmark')
    elif(-1<beta<0):
        print('The asset moves in the opposite direction to the benchmark')
    elif(beta==0):
        print("There is no correlation between the asset's price movement and the market benchmark")
    elif(0<beta<1):
        print("The asset moves in the same direction as the market, but the amount is smaller. An example might be the stock of a company that is not very susceptible to day-to-day fluctuations")
    elif(beta==1):
        print("The asset and the market are moving in the same direction by the same amount")
    elif(beta>1):
        print("The asset moves in the same direction as the market, but the amount is greater. An example might be the stock of a company that is very susceptible to day-to-day market news")

    print('\n\n')
    y = X.pop('asset')
    X = sm.add_constant(X)
    capm_model = sm.OLS(y, X).fit()
    print(capm_model.summary())
