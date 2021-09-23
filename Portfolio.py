import yfinance as yf
import numpy as np
import pandas as pd
import pyfolio as pf
import matplotlib.pyplot as plt
import datetime
import scipy.optimize as sco


def portfolio_report():

    def efficient_frontier(tick_):

        RISKY_ASSETS = tick_
        N_PORTFOLIOS = 10 ** 5

        end = datetime.datetime.now()
        start = datetime.datetime.now() - datetime.timedelta(days=365*2)
        n_assets = len(RISKY_ASSETS)

        prices_df = yf.download(RISKY_ASSETS, start=start, end=end, adjusted=True)
        N_DAYS = prices_df.shape[0]

        def get_portf_rtn(w, avg_rtns):
            return np.sum(avg_rtns * w)
        def get_portf_vol(w, avg_rtns, cov_mat):
            return np.sqrt(np.dot(w.T, np.dot(cov_mat, w)))

        #Define the function calculating the Efficient Frontier:
        def get_efficient_frontier(avg_rtns, cov_mat, rtns_range):
            efficient_portfolios = []
            n_assets = len(avg_returns)
            args = (avg_returns, cov_mat)
            bounds = tuple((0,1) for asset in range(n_assets))
            initial_guess = n_assets * [1. / n_assets, ]
            for ret in rtns_range:
                constraints = ({'type': 'eq', 'fun': lambda x: get_portf_rtn(x, avg_rtns) - ret}, {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
                efficient_portfolio = sco.minimize(get_portf_vol, initial_guess, args=args, method='SLSQP', constraints=constraints, bounds=bounds)
                efficient_portfolios.append(efficient_portfolio)
            return efficient_portfolios

        returns_df = prices_df['Adj Close'].pct_change().dropna()
        avg_returns = returns_df.mean() * N_DAYS
        cov_mat = returns_df.cov() * N_DAYS

        #Simulate random portfolio weights:
        np.random.seed(42)
        weights = np.random.random(size=(N_PORTFOLIOS, n_assets))
        weights /= np.sum(weights, axis=1)[:, np.newaxis]

        #Calculate the portfolio metrics:
        portf_rtns = np.dot(weights, avg_returns)
        portf_vol = []
        for i in range(0, len(weights)):
            portf_vol.append(np.sqrt(np.dot(weights[i].T,
            np.dot(cov_mat, weights[i]))))
        portf_vol = np.array(portf_vol)
        portf_sharpe_ratio = portf_rtns / portf_vol

        #Create a DataFrame containing all the data:
        portf_results_df = pd.DataFrame({'returns': portf_rtns, 'volatility': portf_vol, 'sharpe_ratio': portf_sharpe_ratio})#

        # Define the considered range of returns:
        rtns_range = np.linspace(-0.22, 0.32, 200)
        efficient_portfolios = get_efficient_frontier(avg_returns, cov_mat, rtns_range)
        vols_range = [x['fun'] for x in efficient_portfolios]

        fig, ax = plt.subplots()
        fig.canvas.set_window_title("Portfolio")
        portf_results_df.plot(kind='scatter', x='volatility', y='returns', c='sharpe_ratio', cmap='RdYlGn', edgecolors='black', ax=ax)
        ax.plot(vols_range, rtns_range, 'b--', linewidth=3)
        ax.set(xlabel='Volatility', ylabel='Expected Returns', title='Efficient Frontier')
        plt.show()

        #Identify the minimum volatility portfolio:
        min_vol_ind = np.argmin(vols_range)
        min_vol_portf_rtn = rtns_range[min_vol_ind]
        min_vol_portf_vol = efficient_portfolios[min_vol_ind]['fun']
        min_vol_portf = {'Return': min_vol_portf_rtn, 'Volatility': min_vol_portf_vol, 'Sharpe Ratio': (min_vol_portf_rtn / min_vol_portf_vol)}

        #Print the performance summary:
        print('Minimum volatility portfolio ----')
        print('Performance')
        for index, value in min_vol_portf.items():
            print(f'{index}: {100 * value:.2f}% ', end="", flush=True)
        print('\nWeights')
        for x, y in zip(RISKY_ASSETS,efficient_portfolios[min_vol_ind]['x']):
            print(f'{x}: {100*y:.2f}% ', end="", flush=True)

        plt.show()


    x=int(input("Enter number of companies:"))
    tick_=[]
    for i in range(x):
        val=str(input('Enter ticker:'))
        tick_.append(val)

    tick_=[x.upper().strip() for x in tick_]
    tick_=[x+'.NS' for x in tick_]
    RISKY_ASSETS = tick_
    end = datetime.datetime.now()
    start = datetime.datetime.now() - datetime.timedelta(days=365*2)
    n_assets = len(RISKY_ASSETS)

    prices_df = yf.download(RISKY_ASSETS, start=start, end=end, adjusted=True)
    returns = prices_df['Adj Close'].pct_change().dropna()
    portfolio_weights = n_assets * [1 / n_assets]

    portfolio_returns = pd.Series(np.dot(portfolio_weights, returns.T), index=returns.index)
    pf.create_simple_tear_sheet(portfolio_returns)

    efficient_frontier(tick_)
    plt.show()
