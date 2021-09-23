import pandas as pd
import datetime
import yfinance as yf
import numpy as np
from mlxtend.regressor import StackingCVRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
import plotly.graph_objs as go
import calendar

def modelling(df,data,date_index):

    holidays=['26/01/21','19/02/21','11/03/21','29/03/21','02/04/21','14/04/21','21/04/21','13/05/21','21/07/21',
          '19/08/21','10/09/21','15/10/21','04/11/21','05/11/21','19/11/21']

    holidays=[pd.to_datetime(datetime.datetime.strptime(x, '%d/%m/%y')) for x in holidays]



    X=data.drop('Close(T)',axis=1)
    Y=data['Close(T)']

    stack = StackingCVRegressor(regressors=(
                            XGBRegressor(colsample_bytree=1.0, learning_rate=0.02, max_depth=3, min_child_weight=2, n_estimators=400, nthread=0, objective='reg:linear',silent=0, subsample=0.9),
                            Lasso(alpha=0, copy_X=True, fit_intercept=False, max_iter=10, normalize=True, positive=True, precompute=False, warm_start=True, selection='cyclic'),
                            KNeighborsRegressor(algorithm='auto', leaf_size=5, metric='canberra', n_neighbors=4, p=0, weights='distance', n_jobs=-1)
                                        ),
                            meta_regressor=Lasso(alpha=0, copy_X=True, fit_intercept=False, max_iter=10, normalize=True, positive=True, precompute=False, warm_start=True, selection='cyclic'),
                            cv=10)
    stack.fit(X, Y)

    preds=[]
    for i in range(10):
        vals=list(X.iloc[-1].values)[:-1]
        vals=vals[1:]
        vals.append(list(Y)[-1])
        vals.append(list(X['Volume'])[-1])
        record=pd.DataFrame(vals).T
        record.columns=['f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10']

        pred = stack.predict(record)
        preds.append(pred[0])
        X.loc[len(X)]=vals

    #pred_data=pd.DataFrame(preds,columns=['Close'])
    df_=df.loc[:,'Close']
    df_=df_.reset_index()

    weekend=['Sunday','Saturday']

    for i in range(len(preds)):
        date_=list(df_['Date'])[-1]+datetime.timedelta(days=1)

        if(date_ in holidays):
            date_=date_+datetime.timedelta(days=1)

        day=calendar.day_name[date_.weekday()]
        if(day=='Saturday'):
            date_=date_+datetime.timedelta(days=2)

        df_.loc[len(df_)]=[date_,preds[i]]


    df_.index=df_['Date']
    df_.drop('Date',axis=1,inplace=True)

    return df_

def eda_pred(pred_data,df2):

    df2=df2.iloc[-30:]
    pred_data['Returns']=pred_data['Close']-pred_data['Close'].shift(1)

    print('Next 10 Days Forecasted Data:\n')

    for i in range(1,len(pred_data)):
        print("Close Price on",pred_data.index[i].date(),'will be Rs.',int(pred_data['Close'][i]),'with Return equal to Rs.',round(pred_data['Returns'][i],0))

    print('\nAverage Stock Price for Next 10 days: Rs.',int(pred_data['Close'].mean()))
    print('Overall Return for next days: Rs.',int(pred_data['Returns'].sum()))
    print('Highest Return: Rs.',int(pred_data['Returns'].max()),'at',pred_data[pred_data['Returns']==pred_data['Returns'].max()].index[0].date())
    print('Lowest Return: Rs.',int(pred_data['Returns'].min()),'at',pred_data[pred_data['Returns']==pred_data['Returns'].min()].index[0].date())


    data1=go.Scatter(
            y=pred_data['Close'], x=pred_data.index,mode='lines',
            name='Prediction',
            marker=dict(size=14,color='green',line=dict(width=2)))

    data2=go.Scatter(y=df2['Close'], x=df2.index,mode='lines',name='Actual',
        marker=dict(size=14,color='red',line=dict(width=2)))

    layout=go.Layout(
    title='Close Price',
    xaxis=dict(title='Date'),
    yaxis=dict(title='Prices(Rs)'),
    hovermode='closest')

    data=[data1,data2]

    figure=go.Figure(data=data,layout=layout)
    figure.show()

    data=go.Scatter(
            y=pred_data['Returns'], x=pred_data.index,mode='lines',
            name='Returns')

    layout=go.Layout(
    title='Returns',
    xaxis=dict(title='Date'),
    yaxis=dict(title='Prices(Rs)'),
    hovermode='closest')

    figure=go.Figure(data=data,layout=layout)
    figure.show()

def forecast(tick,name):

    print("----------------------------------------{}-----------------------------------------".format(name))

    end = datetime.datetime.now()
    start = datetime.datetime.now() - datetime.timedelta(days=365)
    df = yf.download(tick,start=start,end=end,progress=False,auto_adjust=True)

    temp=df.loc[:,['Volume']]
    temp=temp.reset_index()

    final=pd.DataFrame(columns=['Date','Close(T-10)','Close(T-9)','Close(T-8)','Close(T-7)','Close(T-6)','Close(T-5)',
                            'Close(T-4)','Close(T-3)','Close(T-2)','Close(T-1)','Close(T)'])
    x=0
    for i in range(10,len(df)):
        ind=df.index[i]
        vals=[]
        vals.append(ind)


        for j in range(x,i):
            vals.append(df['Close'][j])

        vals.append(df['Close'][j+1])
        x=x+1
        final.loc[len(final)]=vals

    final=pd.merge(final,temp,on='Date',how='left')
    date_index=list(final['Date'])
    final.drop('Date',axis=1,inplace=True)

    final_data=modelling(df,final,date_index)
    final_data=final_data.iloc[-11:]

    eda_pred(final_data,df)
