import yfinance as yf
import pandas as pd
import datetime
import numpy as np
from scipy.stats import kurtosis
from scipy.stats import skew
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import scipy.stats as scs
import statsmodels.api as sm
import statsmodels.tsa.api as smt
plt.style.use(['ggplot'])
sns.set()

def outlier(df,var_,t):

    def indentify_outliers(row, n_sigmas=5):
        x = row[var_]
        mu = row['mean']
        sigma = row['std']
        if (x > mu + 3 * sigma) | (x < mu - 3 * sigma):
            return 1
        else:
            return 0

    if(t=='Today'):
        df_rolling = df[[var_]].rolling(window=21).agg(['mean', 'std'])
        df_rolling.columns = df_rolling.columns.droplevel()
    else:
        df_rolling = df[[var_]].rolling(window=5).agg(['mean', 'std'])
        df_rolling.columns = df_rolling.columns.droplevel()

    df_outliers = df.join(df_rolling)

    df_outliers['outlier'] = df_outliers.apply(indentify_outliers,axis=1)
    outliers = df_outliers.loc[df_outliers['outlier'] == 1,[var_]]

    return outliers,df_outliers

def current_price(df,data,var_,name):

    df=df.loc[:,[var_]]
    df['Change']=df[var_]-df[var_].shift(1)
    df['Change_p']=df[var_].pct_change()
    df["returns"]=(df[var_]/df[var_].shift(1))-1
    df["returns"]=df[var_].pct_change(1)
    df=df.iloc[1:]

    cp=df[var_][-1]

    print("---------------------------------{} Current Price Analysis---------------------------------\n".format(name))
    print("Price: Rs",int(cp))
    print("Change:",['Down by Rs' if df['Change'].iloc[-1]<0 else 'Up by Rs'][0],abs(int(df['Change'].iloc[-1])))
    print("Change(%):",['Down by' if df['Change_p'].iloc[-1]<0 else 'Up by'][0],abs(round(df['Change_p'].iloc[-1],4)),'%')

    vals1,vals2=outlier(df,var_,'Today')
    vals=list(vals1[var_])
    vals=[int(i) for i in vals]
    beh=['Outlier' if int(cp) in vals else 'Normal'][0]
    print("Behaviour:",beh)

    val=abs(int(df[var_].iloc[:-1].mean()) - int(cp))
    print("Compared to Today's Average:",['Down by Rs' if val<0 else 'Up by Rs'][0],val)

    val=abs(int(data[var_].iloc[:-5].mean()) - int(cp))
    print("Compared to Last 7 Days Average:",['Down by Rs' if val<0 else 'Up by Rs'][0],val)

    val=abs(int(data[var_].iloc[:-20].mean()) - int(cp))
    print("Compared to Last 30 Days Average:",['Down by Rs' if val<0 else 'Up by Rs'][0],val)

    val=abs(int(data[var_].mean()) - int(cp))
    print("Compared to Last 6 months Average:",['Down by Rs' if val<0 else 'Up by Rs'][0],val)

    print('\n')
    print('-------------------------------------------------------------------------------------------------')
    print('\n')

def today_price(df,data,var_,name):

    df=df.loc[:,[var_]]
    df['Change']=df[var_]-df[var_].shift(1)
    df['Change_p']=df[var_].pct_change()
    df["returns"]=(df[var_]/df[var_].shift(1))-1
    df["returns"]=df[var_].pct_change(1)
    df=df.iloc[1:]

    print("---------------------------------{}-----------------------------------------------------\n".format(name))
    print("---------------------------------Today's Price Analysis---------------------------------\n")
    print("Today's Average Price: Rs",int(df[var_].mean()))
    print('Average Change:',['Down by Rs' if df['Change'].mean()<0 else 'Up by Rs'][0],int(df['Change'].mean()))
    print('Average Change %:',['Down by' if df['Change_p'].mean()<0 else 'Up by'][0],round(df['Change_p'].mean(),4),'%')

    max_diff=data[var_].max()-df[var_].max()
    max_diff_st=['Up by Rs' if df[var_].max() > data[var_].max() else "Down by"][0]
    print("Today's Highest Price: Rs",int(df[var_].max()),'(',max_diff_st,max_diff,')')

    min_diff=data[var_].max()-df[var_].min()
    min_diff_st=['Up by Rs' if df[var_].min() > data[var_].min() else "Down by"][0]
    print("Today's Lowest Price: Rs",int(df[var_].max()),'(',min_diff_st,min_diff,')')

    print('Price Difference from Average Price (Volatility): Rs',int(df[var_].std()))

    print('25th Percentile: Rs.',int(df[var_].describe()['25%']))
    print('50th Percentile: Rs.',int(df[var_].describe()['50%']))
    print('75th Percentile: Rs.',int(df[var_].describe()['75%']))

    print('Average Price for every 5 continious reading: Rs',int(df[var_].rolling(5).mean().mean()))

    vals1,vals2=outlier(df,var_,'Today')
    vals_=list(vals1[var_])
    vals_=[int(i) for i in vals_]
    vals2=list(df[var_])
    vals2=[int(i) for i in vals2]
    vals_1=list(set(vals_).intersection(vals2))

    beh=['Normal' if len(vals_1) == 0 else "Outliers Detected"][0]
    print("Behaviour:",beh)
    if(beh=='Outliers Detected'):
        df_=df[df[var_].apply(lambda x:x in vals_1)]

        for j in range(len(df_)):
            print(df_.index[j],'-----------',df_[var_][j])

    val=abs(int(data[var_].iloc[:-5].mean()) - int(df[var_].mean()))
    print("Compared to Last 7 Days Average:",['Down by Rs' if val<0 else 'Up by Rs'][0],val)

    val=abs(int(data[var_].iloc[:-20].mean()) - int(df[var_].mean()))
    print("Compared to Last 30 Days Average:",['Down by Rs' if val<0 else 'Up by Rs'][0],val)

    val=abs(int(data[var_].mean()) - int(df[var_].mean()))
    print("Compared to Last 6 months Average:",['Down by Rs' if val<0 else 'Up by Rs'][0],val)

    skew_=df[var_].skew()

    if(-0.5<=skew_<=0.5):
        print("Data is Fairly Symmetrical")
    elif((-1<=skew_<-0.5) | (0.5<skew_<=1)):
        print("Data is Moderly Skewed")
    elif((skew_>1) | (skew_<-1)):
        print("Data is Heavnly Skewed")
    else:
        print("Unknown Symmetry")

    kurt=df[var_].kurtosis()
    if(0<=kurt<=0.1):
        print("Data follows Normal Distribution")
    elif(kurt<0):
        print("Data has Light Tails. Data follows Platykurtic Distribution")
    else:
        print("Data has Heavy Tails. Data follows Leptokurtic Distribution")

    print('\n')
    print('-------------------------------------------------------------------------------------------------')
    print('\n')

    d1=go.Scatter(
            y=df[var_], x=df.index,mode='lines',
            name='Up',
            marker=dict(size=14,color='green',line=dict(width=2)))
    d2=go.Scatter(y=df[var_], x=df.index.where(df['returns']<0),mode='lines',name='Down',
        marker=dict(size=14,color='red',line=dict(width=2)))

    layout=go.Layout(
    title='{} Price'.format(var_),
    xaxis=dict(title='Date'),
    yaxis=dict(title='Prices(Rs)'),
    hovermode='closest')

    d=[d1,d2]

    figure=go.Figure(data=d,layout=layout)
    figure.show()

    df['MA15']=df[var_].rolling(3).mean()
    df['MA30']=df[var_].rolling(6).mean()
    df['MA60']=df[var_].rolling(12).mean()


    d2=go.Scatter(
    x=df.index,
    y=df['MA15'],
    mode='lines',
    name='Average 15min Price',
    marker=dict(size=10,color='red',line=dict(width=2)))

    d3=go.Scatter(
        x=df.index,
        y=df['MA30'],
        mode='lines',
        name='Average 30min Price',
        marker=dict(size=10,color='blue',line=dict(width=2)))

    d4=go.Scatter(
        x=df.index,
        y=df['MA60'],
        mode='lines',
        name='Average 60min Price',
        marker=dict(size=10,color='brown',line=dict(width=2)))


    d=[d2,d3,d4]

    layout=go.Layout(
    title='Moving Averages',
    xaxis=dict(title='Date'),
    yaxis=dict(title='Prices(Rs)'),
    hovermode='closest'

    )
    figure=go.Figure(data=d,layout=layout)
    figure.show()

    d=go.Box(y=df[var_], boxpoints="all",name=var_)
    layout=go.Layout(
    title='Price Distribution'.format(var_),
    hovermode='closest')
    figure=go.Figure(data=d,layout=layout)
    figure.show()

    outliers,df_outliers=outlier(df,var_,'Today')
    if(outliers.shape[0]!=0):

        fig, ax = plt.subplots()
        ax.plot(df_outliers.index, df_outliers[var_],color='blue', label='Normal')
        ax.scatter(outliers.index, outliers[var_],color='red', label='Anomaly')
        ax.set_title("Outliers Detection")
        ax.legend(loc='lower right')

    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    fig.canvas.set_window_title("{} Today's Data Distribution".format(name))

    # Calculate the normal Probability Density Function (PDF) using the mean and standard deviation of the observed returns
    r_range = np.linspace(min(df[var_]), max(df[var_]), num=1000)
    mu = df[var_].mean()
    sigma = df[var_].std()
    norm_pdf = scs.norm.pdf(r_range, loc=mu, scale=sigma)

    # histogram
    sns.distplot(df[var_], kde=False, norm_hist=True, ax=ax[0])
    ax[0].set_title('Distribution of {} Price'.format(var_), fontsize=16)
    ax[0].plot(r_range, norm_pdf, 'g', lw=2,label=f'N({mu:.2f}, {sigma**2:.4f})')
    ax[0].legend(loc='upper left');

    # Q-Q plot
    qq = sm.qqplot(df[var_].values, line='s', ax=ax[1])
    ax[1].set_title('Q-Q plot', fontsize = 16)
    plt.show()

def overall(df,var_,name):

    df=df.loc[:,[var_]]
    df['Change']=df[var_]-df[var_].shift(1)
    df['Change_p']=df[var_].pct_change()
    df["returns"]=(df[var_]/df[var_].shift(1))-1
    df["returns"]=df[var_].pct_change(1)
    df=df.iloc[1:]

    print("---------------------------------{}-----------------------------------------------------\n".format(name))
    print("-----------------------------Last 6 months Price Analysis-------------------------------\n")

    print('Average Price: Rs',int(df[var_].mean()))
    print("Highest Price: Rs",df[var_].max())
    print("Lowest Price: Rs",df[var_].min())
    print('Average Change:',['Down by Rs' if df['Change'].mean()<0 else 'Up by Rs'][0],int(df['Change'].mean()))
    print('Average Change %:',['Down by' if df['Change_p'].mean()<0 else 'Up by'][0],round(df['Change_p'].mean(),4),'%')
    print('Price Difference from Average Price (Volatility): Rs',int(df[var_].std()))
    print('25th Percentile: Rs.',int(df[var_].describe()['25%']))
    print('50th Percentile: Rs.',int(df[var_].describe()['50%']))
    print('75th Percentile: Rs.',int(df[var_].describe()['75%']))
    print('Week Average: Rs',int(df[var_].rolling(7).mean().mean()))
    print('15 Days Average: Rs',int(df[var_].rolling(15).mean().mean()))
    print('Month Average: Rs',int(df[var_].rolling(30).mean().mean()))
    print('3 Months Average: Rs',int(df[var_].rolling(60).mean().mean()))

    vals1,vals2=outlier(df,var_,'Overall')
    vals_=list(vals1[var_])
    vals_=[int(i) for i in vals_]
    vals2=list(df[var_])
    vals2=[int(i) for i in vals2]
    vals_1=list(set(vals_).intersection(vals2))

    beh=['Normal' if len(vals_1) == 0 else "Outliers Detected"][0]
    print("Behaviour:",beh)
    if(beh=='Outliers Detected'):
        df_=df[df[var_].apply(lambda x:x in vals_1)]

        for j in range(len(df_)):
            print(df_.index[j],'-----------',df_[var_][j])

    skew_=df[var_].skew()

    if(-0.5<=skew_<=0.5):
        print("Data is Fairly Symmetrical")
    elif((-1<=skew_<-0.5) | (0.5<skew_<=1)):
        print("Data is Moderly Skewed")
    elif((skew_>1) | (skew_<-1)):
        print("Data is Heavnly Skewed")
    else:
        print("Unknown Symmetry")

    kurt=df[var_].kurtosis()
    if(0<=kurt<=0.1):
        print("Data follows Normal Distribution")
    elif(kurt<0):
        print("Data has Light Tails. Data follows Platykurtic Distribution")
    else:
        print("Data has Heavy Tails. Data follows Leptokurtic Distribution")

    df['Week']=0
    for i in range(len(df)):
        y=df.index[i]
        x=int(y.day)

        if(x<=7):
            df.loc[y,"Week"]='Week 1'
        elif((x>7) & (x<=14)):
            df.loc[y,"Week"]='Week 2'
        elif((x>14) & (x<=21)):
            df.loc[y,"Week"]='Week 3'
        elif((x>21) & (x<=28)):
            df.loc[y,"Week"]='Week 4'
        elif(x>28):
            df.loc[y,"Week"]='Week 5'

    week_df=pd.DataFrame(df.groupby('Week')[var_].mean()).reset_index()
    week_df=week_df[week_df['Week']!='Week 5']
    by_week=df.resample('W').mean()
    by_month=df.resample('M').mean()

    weeks=['Week 1','Week 2','Week 3','Week 4']
    print('\nAverage Price By Week')
    for i in weeks:
        record=df[df['Week']==i]
        print(i,': Rs.',round(record[var_].mean()))


    d1=go.Scatter(
                y=df[var_], x=df.index,mode='lines',
                name='Up'.format(var_),
                marker=dict(size=14,color='green',line=dict(width=2)))
    d2=go.Scatter(y=df[var_], x=df.index.where(df['returns']<0),mode='lines',name='Down'.format(var_),
        marker=dict(size=14,color='red',line=dict(width=2)))

    layout=go.Layout(
    title='{} Price'.format(var_),
    xaxis=dict(title='Date'),
    yaxis=dict(title='Prices(Rs)'),
    hovermode='closest')

    d=[d1,d2]

    figure=go.Figure(data=d,layout=layout)
    figure.show()

    df['MA7']=df[var_].rolling(7).mean()
    df['MA20']=df[var_].rolling(20).mean()
    df['MA30']=df[var_].rolling(30).mean()

    d2=go.Scatter(
        x=df.index,
        y=df['MA7'],
        mode='lines',
        name='7-Days Average',
        marker=dict(size=10,color='red',line=dict(width=2)))

    d3=go.Scatter(
        x=df.index,
        y=df['MA20'],
        mode='lines',
        name='20-Days Average',
        marker=dict(size=10,color='blue',line=dict(width=2)))

    d4=go.Scatter(
        x=df.index,
        y=df['MA30'],
        mode='lines',
        name='30-Days Average',
        marker=dict(size=10,color='brown',line=dict(width=2)))

    d=[d2,d3,d4]

    layout=go.Layout(
    title='Moving Averages',
    xaxis=dict(title='Date'),
    yaxis=dict(title='Prices(Rs)'),
    hovermode='closest')

    figure=go.Figure(data=d,layout=layout)
    figure.show()

    d=go.Box(y=df[var_], boxpoints="all",name=var_)
    layout=go.Layout(
    title='{} Price Distribution'.format(var_),
    hovermode='closest')
    figure=go.Figure(data=d,layout=layout)
    figure.show()

    d=go.Bar(
            x=week_df['Week'],
            y=week_df[var_],
            name='Average Price Per Week')

    layout=go.Layout(
    title='Average Price Per Week',
    xaxis=dict(title='Week'),
    yaxis=dict(title='Prices(Rs)'),
    hovermode='closest')
    figure=go.Figure(data=d,layout=layout)
    figure.show()

    d1=go.Scatter(
        x=by_week.index,
        y=by_week[var_],
        mode='lines',
        name='Weekly {} Price'.format(var_),
        marker=dict(size=10,color='blue',line=dict(width=2)))

    d2=go.Scatter(
        x=by_month.index,
        y=by_month[var_],
        mode='lines',
        name='Monthly {} Price'.format(var_),
        marker=dict(size=10,color='brown',line=dict(width=2)))


    d=[d1,d2]

    layout=go.Layout(
    title='Weekly/Monthly {} Price Performance'.format(var_),
    xaxis=dict(title='Date'),
    yaxis=dict(title='Prices(Rs)'),
    hovermode='closest'

    )
    figure=go.Figure(data=d,layout=layout)
    figure.show()

    outliers,df_outliers=outlier(df,var_,'Overall')

    if(outliers.shape[0]!=0):
        fig, ax = plt.subplots()
        ax.plot(df_outliers.index, df_outliers.simple_rtn,color='blue', label='Normal')
        ax.scatter(outliers.index, outliers.simple_rtn,color='red', label='Anomaly')
        ax.set_title("Outliers")
        ax.legend(loc='lower right')

    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    fig.canvas.set_window_title("{} Overall Data Distribution".format(name))

    # Calculate the normal Probability Density Function (PDF) using the mean and standard deviation of the observed returns
    r_range = np.linspace(min(df[var_]), max(df[var_]), num=1000)
    mu = df[var_].mean()
    sigma = df[var_].std()
    norm_pdf = scs.norm.pdf(r_range, loc=mu, scale=sigma)

    # histogram
    sns.distplot(df[var_], kde=False, norm_hist=True, ax=ax[0])
    ax[0].set_title('Distribution of {} Price'.format(var_), fontsize=16)
    ax[0].plot(r_range, norm_pdf, 'g', lw=2,label=f'N({mu:.2f}, {sigma**2:.4f})')
    ax[0].legend(loc='upper left');

    # Q-Q plot
    qq = sm.qqplot(df[var_].values, line='s', ax=ax[1])
    ax[1].set_title('Q-Q plot', fontsize = 16)
    plt.show()

def returns_today(df,data,name):

    df=df.loc[:,['Close']]
    df['Returns']=df['Close']-df['Close'].shift(1)
    df['Returns_Change']=df['Returns'].pct_change()
    df.fillna(0,inplace=True)
    df=df.replace([np.inf, -np.inf], np.nan).dropna(axis=0)

    data=data.loc[:,['Close']]
    data['Returns']=data['Close']-data['Close'].shift(1)
    data['Returns_Change']=data['Returns'].pct_change()
    data.fillna(0,inplace=True)
    data=data.replace([np.inf, -np.inf], np.nan).dropna(axis=0)

    print("|------------- {} -------------|\n".format(name))
    print("|------------- Today's Returns Price Analysis -------------|\n")

    print("Total Return Untill Now:",['Loss of Rs' if df['Returns'].sum()<0 else 'Profit of Rs'][0],abs(int(df['Returns'].sum())))
    print("Last 15min Return:",['Loss of Rs' if df['Returns'].iloc[-1]<0 else 'Profit of Rs'][0],int(df['Returns'].iloc[-1]))
    print("Average Return:",['Loss of Rs' if df['Returns'].mean()<0 else 'Profit of Rs'][0],round(df['Returns'].mean(),4))
    print("Average Return Change %:",['Down By' if df['Returns_Change'].mean()<0 else 'Up by'][0],round(df['Returns_Change'].mean(),4))
    print("Highest Return:",['Loss of Rs' if df['Returns'].max()<0 else 'Profit of Rs'][0],int(df['Returns'].max()))
    print("Lowest Return:",['Loss of Rs' if df['Returns'].min()<0 else 'Profit of Rs'][0],abs(int(df['Returns'].min())))
    print("60 Min Average: Rs",round(df['Returns'].rolling(4).mean().mean(),4))
    print("45 Min Average: Rs",round(df['Returns'].rolling(3).mean().mean(),4))
    print("30 Min Average: Rs",round(df['Returns'].rolling(2).mean().mean(),4))
    print("Difference to last 6 months average returns:",['Down By Rs' if data['Returns'].mean()-df['Returns'].mean()<0 else 'Up by Rs'][0],abs(int(data['Returns'].mean()-df['Returns'].mean())))
    print("Difference to last 3 months average returns:",['Down By Rs' if data['Returns'].iloc[:-90].mean()-df['Returns'].mean()<0 else 'Up by Rs'][0],abs(int(data['Returns'].iloc[:-90].mean()-df['Returns'].mean())))
    print("Difference to last 1 month average returns:",['Down By Rs' if data['Returns'].iloc[:-30].mean()-df['Returns'].mean()<0 else 'Up by Rs'][0],abs(int(data['Returns'].iloc[:-30].mean()-df['Returns'].mean())))

    skew_=df['Returns'].skew()

    if(-0.5<=skew_<=0.5):
        print("Data is Fairly Symmetrical")
    elif((-1<=skew_<-0.5) | (0.5<skew_<=1)):
        print("Data is Moderly Skewed")
    elif((skew_>1) | (skew_<-1)):
        print("Data is Heavnly Skewed")
    else:
        print("Unknown Symmetry")

    kurt=df['Returns'].kurtosis()
    if(0<=kurt<=0.1):
        print("Data follows Normal Distribution")
    elif(kurt<0):
        print("Data has Light Tails. Data follows Platykurtic Distribution")
    else:
        print("Data has Heavy Tails. Data follows Leptokurtic Distribution")


    d1=go.Scatter(
                y=df["Returns"], x=df.index,mode='lines',
                name='Up',
                marker=dict(size=14,color='green',line=dict(width=2)))
    d2=go.Scatter(y=df["Returns"], x=df.index.where(df['Returns']<0),mode='lines',name='Down',
        marker=dict(size=14,color='red',line=dict(width=2)))

    layout=go.Layout(
    title="{} Today's Returns".format(name),
    xaxis=dict(title='Date'),
    yaxis=dict(title='Prices(Rs)'),
    hovermode='closest')

    d=[d1,d2]

    figure=go.Figure(data=d,layout=layout)
    figure.show()

    d1=go.Scatter(
                y=df["Returns_Change"], x=df.index,mode='lines',
                name='Up',
                marker=dict(size=14,color='green',line=dict(width=2)))
    d2=go.Scatter(y=df["Returns_Change"], x=df.index.where(df['Returns_Change']<0),mode='lines',name='Down',
        marker=dict(size=14,color='red',line=dict(width=2)))

    layout=go.Layout(
    title='Returns Change(%)',
    xaxis=dict(title='Date'),
    yaxis=dict(title='Prices(Rs)'),
    hovermode='closest')

    d=[d1,d2]

    figure=go.Figure(data=d,layout=layout)
    figure.show()

    df['MA60']=df["Returns"].rolling(10).mean()
    df['MA45']=df["Returns"].rolling(9).mean()
    df['MA30']=df["Returns"].rolling(6).mean()
    df['MA15']=df["Returns"].rolling(3).mean()

    d1=go.Scatter(
                y=df["MA60"], x=df.index,mode='lines',
                name='60 Minute Average Returns',
                marker=dict(size=14,color='green',line=dict(width=2)))

    d2=go.Scatter(
                y=df["MA45"], x=df.index,mode='lines',
                name='45 Minute Average Returns',
                marker=dict(size=14,color='red',line=dict(width=2)))

    d3=go.Scatter(
                y=df["MA30"], x=df.index,mode='lines',
                name='30 Minute Average Returns',
                marker=dict(size=14,color='blue',line=dict(width=2)))

    d4=go.Scatter(
                y=df["MA15"], x=df.index,mode='lines',
                name='15 Minute Average Returns',
                marker=dict(size=14,color='blue',line=dict(width=2)))

    layout=go.Layout(
    title='Moving Average Of Returns',
    xaxis=dict(title='Date'),
    yaxis=dict(title='Prices(Rs)'),
    hovermode='closest')
    d=[d1,d2,d3,d4]

    figure=go.Figure(data=d,layout=layout)
    figure.show()

    d=go.Box(y=df["Returns"], boxpoints="all",name="Returns")
    layout=go.Layout(
    title='Returns Distribution',
    hovermode='closest')
    figure=go.Figure(data=d,layout=layout)
    figure.show()

    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    fig.canvas.set_window_title("{} Today's Return Distribution".format(name))

    # Calculate the normal Probability Density Function (PDF) using the mean and standard deviation of the observed returns
    r_range = np.linspace(min(df["Returns"]), max(df["Returns"]), num=1000)
    mu = df["Returns"].mean()
    sigma = df["Returns"].std()
    norm_pdf = scs.norm.pdf(r_range, loc=mu, scale=sigma)

    # histogram
    sns.distplot(df["Returns"], kde=False, norm_hist=True, ax=ax[0])
    ax[0].set_title('Distribution of Returns Price', fontsize=16)
    ax[0].plot(r_range, norm_pdf, 'g', lw=2,label=f'N({mu:.2f}, {sigma**2:.4f})')
    ax[0].legend(loc='upper left');

    # Q-Q plot
    qq = sm.qqplot(df["Returns"].values, line='s', ax=ax[1])
    ax[1].set_title('Q-Q plot', fontsize = 16)
    plt.show()

def overall_return(data,name):

    data=data.loc[:,['Close']]
    data['Returns']=data['Close']-data['Close'].shift(1)
    data['Returns_Change']=data['Returns'].pct_change()
    data.fillna(0,inplace=True)
    data=data.replace([np.inf, -np.inf], np.nan).dropna(axis=0)

    print("|------------- {} -------------|\n".format(name))
    print("|------------- Last 6 Months Returns Analysis -------------|\n")

    print("Average Return: Rs",round(data['Returns'].mean(),4))
    print("Average Return Change %:",['Down By' if data['Returns_Change'].mean()<0 else 'Up by'][0],round(data['Returns_Change'].mean(),4))
    print("Last Return:",['Loss of Rs' if data['Returns'].iloc[-1]<0 else 'Profit of Rs'][0],abs(int(data['Returns'].iloc[-1])))
    print("Highest Return: Rs",int(data['Returns'].max()))
    print("Lowest Return: Rs",abs(int(data['Returns'].min())))
    print("Average Week Return: Rs",abs(round(data['Returns'].rolling(5).mean().mean(),4)))
    print("Average Month Return: Rs",abs(round(data['Returns'].rolling(22).mean().mean(),4)))

    skew_=data['Returns'].skew()

    if(-0.5<=skew_<=0.5):
        print("Data is Fairly Symmetrical")
    elif((-1<=skew_<-0.5) | (0.5<skew_<=1)):
        print("Data is Moderly Skewed")
    elif((skew_>1) | (skew_<-1)):
        print("Data is Heavnly Skewed")
    else:
        print("Unknown Symmetry")

    kurt=data['Returns'].kurtosis()
    if(0<=kurt<=0.1):
        print("Data follows Normal Distribution")
    elif(kurt<0):
        print("Data has Light Tails. Data follows Platykurtic Distribution")
    else:
        print("Data has Heavy Tails. Data follows Leptokurtic Distribution")

    data['Week']=0
    for i in range(len(data)):
        y=data.index[i]
        x=int(y.day)

        if(x<=7):
            data.loc[y,"Week"]="Week 1"
        elif((x>7) & (x<=14)):
            data.loc[y,"Week"]="Week 2"
        elif((x>14) & (x<=21)):
            data.loc[y,"Week"]="Week 3"
        elif((x>21) & (x<=28)):
            data.loc[y,"Week"]="Week 4"
        elif(x>28):
            data.loc[y,"Week"]="Week 5"


    week_df=pd.DataFrame(data.groupby('Week')["Returns"].mean()).reset_index()
    week_df=week_df[week_df['Week']!='Week 5']
    by_week=data.resample('W').mean()
    by_month=data.resample('M').mean()

    d1=go.Scatter(
                y=data["Returns"], x=data.index,mode='lines',
                name='Up',
                marker=dict(size=14,color='green',line=dict(width=2)))
    d2=go.Scatter(y=data["Returns"], x=data.index.where(data['Returns']<0),mode='lines',name='Down',
        marker=dict(size=14,color='red',line=dict(width=2)))

    layout=go.Layout(
    title='6 Months Returns',
    xaxis=dict(title='Date'),
    yaxis=dict(title='Prices(Rs)'),
    hovermode='closest')

    d=[d1,d2]

    figure=go.Figure(data=d,layout=layout)
    figure.show()

    d1=go.Scatter(
                y=data["Returns_Change"], x=data.index,mode='lines',
                name='Up',
                marker=dict(size=14,color='green',line=dict(width=2)))
    d2=go.Scatter(y=data["Returns_Change"], x=data.index.where(data['Returns_Change']<0),mode='lines',name='Down',
        marker=dict(size=14,color='red',line=dict(width=2)))

    layout=go.Layout(
    title='6 Months Returns Change(%)',
    xaxis=dict(title='Date'),
    yaxis=dict(title='Prices(Rs)'),
    hovermode='closest')

    d=[d1,d2]

    figure=go.Figure(data=d,layout=layout)
    figure.show()

    data['MA30']=data["Returns"].rolling(22).mean()
    data['MA14']=data["Returns"].rolling(10).mean()
    data['MA7']=data["Returns"].rolling(5).mean()

    d1=go.Scatter(
                y=data["MA30"], x=data.index,mode='lines',
                name='1 Month Average Returns',
                marker=dict(size=14,color='green',line=dict(width=2)))

    d2=go.Scatter(
                y=data["MA14"], x=data.index,mode='lines',
                name='2 Weeks Average Returns',
                marker=dict(size=14,color='red',line=dict(width=2)))

    d3=go.Scatter(
                y=data["MA7"], x=data.index,mode='lines',
                name='1 Week Average Returns',
                marker=dict(size=14,color='blue',line=dict(width=2)))

    layout=go.Layout(
    title='Moving Average Returns',
    xaxis=dict(title='Date'),
    yaxis=dict(title='Prices(Rs)'),
    hovermode='closest')

    d=[d1,d2,d3]

    figure=go.Figure(data=d,layout=layout)
    figure.show()

    d=go.Box(y=data["Returns"], boxpoints="all",name="Returns")
    layout=go.Layout(
    title='Returns Distribution',
    hovermode='closest')
    figure=go.Figure(data=d,layout=layout)
    figure.show()

    d=go.Bar(
        x=week_df['Week'],
        y=week_df["Returns"],
        name='Return')

    layout=go.Layout(
    title='Average Returns Per Week',
    xaxis=dict(title='Week'),
    yaxis=dict(title='Prices(Rs)'),
    hovermode='closest')
    figure=go.Figure(data=d,layout=layout)
    figure.show()

    d1=go.Scatter(
        x=by_week.index,
        y=by_week["Returns"],
        mode='lines',
        name='Weekly Returns',
        marker=dict(size=10,color='blue',line=dict(width=2)))

    d2=go.Scatter(
        x=by_month.index,
        y=by_month["Returns"],
        mode='lines',
        name='Monthly Returns',
        marker=dict(size=10,color='brown',line=dict(width=2)))

    d=[d1,d2]

    layout=go.Layout(
    title='Weekly/Monthly Returns',
    xaxis=dict(title='Date'),
    yaxis=dict(title='Prices(Rs)'),
    hovermode='closest'

    )
    figure=go.Figure(data=d,layout=layout)
    figure.show()

    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    fig.canvas.set_window_title("{} 6 Months Return Distribution".format(name))

    # Calculate the normal Probability Density Function (PDF) using the mean and standard deviation of the observed returns
    r_range = np.linspace(min(data["Returns"]), max(data["Returns"]), num=1000)
    mu = data["Returns"].mean()
    sigma = data["Returns"].std()
    norm_pdf = scs.norm.pdf(r_range, loc=mu, scale=sigma)

    # histogram
    sns.distplot(data["Returns"], kde=False, norm_hist=True, ax=ax[0])
    ax[0].set_title('Distribution of Returns Price', fontsize=16)
    ax[0].plot(r_range, norm_pdf, 'g', lw=2,label=f'N({mu:.2f}, {sigma**2:.4f})')
    ax[0].legend(loc='upper left');

    # Q-Q plot
    qq = sm.qqplot(data["Returns"].values, line='s', ax=ax[1])
    ax[1].set_title('Q-Q plot', fontsize = 16)
    plt.show()

def all_var(df,name):

    df['Diff(Open_Close)']=df['Close']-df['Open']
    df['Diff(High_Low)']=df['High']-df['Low']

    print('Average Difference Between Close And Open Price(Per Day): Rs.',round(df['Diff(Open_Close)'].mean(),2))
    print('Average Difference Between Highest And Lowest Price(Per Day): Rs.',round(df['Diff(High_Low)'].mean(),2))

    fig = go.Figure(data=[go.Candlestick(x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'])])
    fig.update_layout(
        title= {
            'text':'OHLC Chart' ,
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
          font=dict(
            family="Courier New, monospace",
            size=20,
            color="#7f7f7f"
        )
        )
    fig.show()

    d1=go.Scatter(
    x=df.index,
    y=df['Open'],
    mode='lines',
    name='Open Price',
    marker=dict(size=14,color='rgb(0,0,100)',line=dict(width=2)))

    d2=go.Scatter(
    x=df.index,
    y=df['Close'],
    mode='lines',
    name='Close Price',
    marker=dict(size=14,color='rgb(0,100,0)',line=dict(width=2)))

    d3=go.Scatter(
    x=df.index,
    y=df['High'],
    mode='lines',
    name='High Price',
    marker=dict(size=14,color='rgb(100,0,0)',line=dict(width=2)))

    d4=go.Scatter(
    x=df.index,
    y=df['Low'],
    mode='lines',
    name='Low Price',
    marker=dict(size=14,color='rgb(50,0,0)',line=dict(width=2)))

    d=[d1,d2,d3,d4]

    layout=go.Layout(
    title='{} Prices'.format(name),
    xaxis=dict(title='Date'),
    yaxis=dict(title='Prices'),
    hovermode='closest')
    figure=go.Figure(data=d,layout=layout)
    figure.show()


    d=go.Scatter( x=df.index, y=df['Diff(Open_Close)'], mode='lines', name='Open/Close Price Difference',
        marker=dict(size=14,color='rgb(0,0,100)',line=dict(width=2)))

    layout=go.Layout(
    title='Open/Close Price Differences',
    xaxis=dict(title='Date'),
    yaxis=dict(title='Prices'),
    hovermode='closest' )
    figure=go.Figure(data=d,layout=layout)
    figure.show()

    d=go.Scatter(
    x=df.index,
    y=df['Diff(High_Low)'],
    mode='lines',
    name='Open/Close Price Difference',
    marker=dict(size=14,color='rgb(0,0,100)',line=dict(width=2)))

    layout=go.Layout(
    title='High/Low Price Differences',
    xaxis=dict(title='Date'),
    yaxis=dict(title='Prices'),
    hovermode='closest')
    figure=go.Figure(data=d,layout=layout)
    figure.show()

def volume_eda(df,name):

    by_week=df.resample('W').mean()
    by_month=df.resample('M').mean()

    print('Total Stockes Traded(In last 6 months):',int(df['Volume'].sum()/10000000),'Crores')
    print('Average Stocks Traded Daily:',int(df['Volume'].mean()/1000000),'Lakhs')
    print('Average Change in traded stocks:',int((df['Volume'].pct_change().mean()/100)*df['Volume'].mean()))

    d=go.Scatter(
            y=df['Volume'], x=df.index,mode='lines',
            name='Volume',
            marker=dict(size=14,color='green',line=dict(width=2)))

    layout=go.Layout(
    title='Stocks Traded',
    xaxis=dict(title='Date'),
    yaxis=dict(title='Prices(Rs)'),
    hovermode='closest')

    figure=go.Figure(data=d,layout=layout)
    figure.show()

    d1=go.Scatter(
        x=by_week.index,
        y=by_week['Volume'],
        mode='lines',
        name='Weekly Stocks Traded',
        marker=dict(size=10,color='blue',line=dict(width=2)))

    d2=go.Scatter(
        x=by_month.index,
        y=by_month['Volume'],
        mode='lines',
        name='Monthly Stocks Traded',
        marker=dict(size=10,color='brown',line=dict(width=2)))

    d=[d1,d2]

    layout=go.Layout(
    title='Weekly/Monthly Stocks Traded',
    xaxis=dict(title='Date'),
    yaxis=dict(title='Prices(Rs)'),
    hovermode='closest')
    figure=go.Figure(data=d,layout=layout)
    figure.show()

    d=go.Box(y=df['Volume'], boxpoints="all",name='Volume')
    layout=go.Layout(
    title='Volume Distribution',
    hovermode='closest')
    figure=go.Figure(data=d,layout=layout)
    figure.show()

    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    fig.canvas.set_window_title("{} Volume Distribution".format(name))

    # Calculate the normal Probability Density Function (PDF) using the mean and standard deviation of the observed returns
    r_range = np.linspace(min(df['Volume']), max(df['Volume']), num=1000)
    mu = df['Volume'].mean()
    sigma = df['Volume'].std()
    norm_pdf = scs.norm.pdf(r_range, loc=mu, scale=sigma)

    # histogram
    sns.distplot(df['Volume'], kde=False, norm_hist=True, ax=ax[0])
    ax[0].set_title('Distribution of Volume', fontsize=16)
    ax[0].plot(r_range, norm_pdf, 'g', lw=2,label=f'N({mu:.2f}, {sigma**2:.4f})')
    ax[0].legend(loc='upper left');

    # Q-Q plot
    qq = sm.qqplot(df['Volume'].values, line='s', ax=ax[1])
    ax[1].set_title('Q-Q plot', fontsize = 16)
    plt.show()
