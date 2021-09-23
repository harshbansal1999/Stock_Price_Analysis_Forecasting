import pandas as pd
import yfinance as yf
import statsmodels.api as sm
import datetime
from arch import arch_model
import matplotlib.pyplot as plt

def garch(tick,name):

    RISKY_ASSET = tick
    end = datetime.datetime.now()
    start = datetime.datetime.now() - datetime.timedelta(days=90)
    df = yf.download(RISKY_ASSET, start=start, end=end, auto_adjust=True)

    print("------------------------------------- Analysis Based on Last 2 months Data -------------------------------------")
    print("------------------------------------- {} -------------------------------------\n".format(name))

    returns = 100 * df['Close'].pct_change().dropna()
    returns.name = 'asset_returns'

    model = arch_model(returns, mean='Zero', vol='GARCH', p=1, o=0,  q=1)
    #Estimate the model and print the summary:
    model_fitted = model.fit(disp='off')
    print(model_fitted.summary())

    model_fitted.plot(annualize='D')
    plt.show()
