import numpy as np
import pandas as pd
import yfinance as yf
import datetime
import matplotlib.pyplot as plt
from scipy import stats

def monte_carlo_simulation(tick,name):

    print("------------------------------------- {} -------------------------------------\n".format(name))
    print("------------------------------------- Analysis Based on Last 1 Year Data -------------------------------------")

    RISKY_ASSET = tick
    end = datetime.datetime.now()
    start = datetime.datetime.now() - datetime.timedelta(days=365)

    df = yf.download(RISKY_ASSET, start=start, end=end, auto_adjust=True, progress=False)

    adj_close = df['Close']
    returns = adj_close.pct_change().dropna()

    train = returns[:-25]
    test = returns[-25:]

    T = len(test)
    N = len(test)
    S_0 = adj_close[train.index[-1]]
    N_SIM = 100
    mu = train.mean()
    sigma = train.std()


    def simulate_gbm(s_0, mu, sigma, n_sims, T, N):
        dt = T/N
        dW = np.random.normal(scale = np.sqrt(dt),
        size=(n_sims, N))
        W = np.cumsum(dW, axis=1)
        time_step = np.linspace(dt, T, N)
        time_steps = np.broadcast_to(time_step, (n_sims, N))
        S_t = s_0 * np.exp((mu - 0.5 * sigma ** 2) * time_steps
        + sigma * W)
        S_t = np.insert(S_t, 0, s_0, axis=1)
        return S_t


    gbm_simulations = simulate_gbm(S_0, mu, sigma, N_SIM, T, N)

    LAST_TRAIN_DATE = train.index[-1].date()
    FIRST_TEST_DATE = test.index[0].date()
    LAST_TEST_DATE = test.index[-1].date()
    PLOT_TITLE = (f'{name} Simulation 'f'({FIRST_TEST_DATE}:{LAST_TEST_DATE})')

    selected_indices = adj_close[LAST_TRAIN_DATE:LAST_TEST_DATE].index
    index = [date.date() for date in selected_indices]
    gbm_simulations_df = pd.DataFrame(np.transpose(gbm_simulations),index=index)

    ax = gbm_simulations_df.plot(alpha=0.2, legend=False,figsize=(20,10))
    line_1, = ax.plot(index, gbm_simulations_df.mean(axis=1),color='red')
    line_2, = ax.plot(index, adj_close[LAST_TRAIN_DATE:LAST_TEST_DATE],color='blue')
    ax.set_title(PLOT_TITLE, fontsize=16)
    ax.legend((line_1, line_2), ('mean', 'actual'))

    rets_1 = (df['Close']/df['Close'].shift(1))-1
    mean = np.mean(rets_1)
    std = np.std(rets_1)
    Z_99 = stats.norm.ppf(1-0.99)
    price = df.iloc[-1]['Close']
    ParamVAR = price*Z_99*std
    HistVAR = price*np.percentile(rets_1.dropna(), 1)
    print("Value at Risk ----------------------------\n")
    print('Parametric VAR is {0:.3f} and Historical VAR is {1:.3f}'.format(ParamVAR, HistVAR))
    np.random.seed(42)
    n_sims = 1000000
    sim_returns = np.random.normal(mean, std, n_sims)
    SimVAR = price*np.percentile(sim_returns, 1)
    print('Simulated VAR is ', SimVAR)

    plt.show()
