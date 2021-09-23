# Import libraries
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
import os
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')
import plotly.graph_objs as go
from wordcloud import WordCloud
finwiz_url = 'https://finviz.com/quote.ashx?t='


def news(tick):
    news_tables = {}
    tickers = [tick]

    for ticker in tickers:
        url = 'https://finviz.com/quote.ashx?t=' + ticker.lower().strip()[:-3]
        req = Request(url=url,headers={'user-agent': 'my-app/0.0.1'})
        response = urlopen(req)
        # Read the contents of the file into 'html'
        html = BeautifulSoup(response)
        # Find 'news-table' in the Soup and load it into 'news_table'
        news_table = html.find(id='news-table')
        # Add the table to our dictionary
        news_tables[ticker] = news_table

    parsed_news = []

    # Iterate through the news
    for file_name, news_table in news_tables.items():
        # Iterate through all tr tags in 'news_table'
        for x in news_table.findAll('tr'):
            # read the text from each tr tag into text
            # get text from a only
            text = x.a.get_text()
            # splite text in the td tag into a list
            date_scrape = x.td.text.split()
            # if the length of 'date_scrape' is 1, load 'time' as the only element

            if len(date_scrape) == 1:
                time = date_scrape[0]

            # else load 'date' as the 1st element and 'time' as the second
            else:
                date = date_scrape[0]
                time = date_scrape[1]
            # Extract the ticker from the file name, get the string up to the 1st '_'
            ticker = file_name.split('_')[0]

            # Append ticker, date, time and headline as a list to the 'parsed_news' list
            parsed_news.append([ticker, date, time, text])

    vader = SentimentIntensityAnalyzer()

    # Set column names
    columns = ['ticker', 'date', 'time', 'headline']

    # Convert the parsed_news list into a DataFrame called 'parsed_and_scored_news'
    parsed_and_scored_news = pd.DataFrame(parsed_news, columns=columns)

    # Iterate through the headlines and get the polarity scores using vader
    scores = parsed_and_scored_news['headline'].apply(vader.polarity_scores).tolist()

    # Convert the 'scores' list of dicts into a DataFrame
    scores_df = pd.DataFrame(scores)

    # Join the DataFrames of the news and the list of dicts
    parsed_and_scored_news = parsed_and_scored_news.join(scores_df, rsuffix='_right')

    # Convert the date column from string to datetime
    parsed_and_scored_news['date'] = pd.to_datetime(parsed_and_scored_news.date).dt.date
    yr=datetime.datetime.now().year
    parsed_and_scored_news=parsed_and_scored_news[parsed_and_scored_news['date'].apply(lambda x:x.year==yr)]

    parsed_and_scored_news['Sentiment']=''
    for i in range(len(parsed_and_scored_news)):
        x=parsed_and_scored_news['compound'][i]

        if(x<0):
            parsed_and_scored_news['Sentiment'][i]='Negative'
        elif(0<=x<=0.4):
            parsed_and_scored_news['Sentiment'][i]='Neutral'
        else:
            parsed_and_scored_news['Sentiment'][i]='Positive'

    parsed_and_scored_news=parsed_and_scored_news.loc[:,['date','headline','Sentiment']]

    for i in range(len(parsed_and_scored_news)):
        print(parsed_and_scored_news['date'][i],'\t',parsed_and_scored_news['headline'][i],'\t',parsed_and_scored_news['Sentiment'][i])

    x=parsed_and_scored_news['Sentiment'].value_counts()
    data=[go.Pie(
            labels=list(x.index),
            values=list(x.values))]

    layout=go.Layout(
    title='Sentiments',
    xaxis=dict(title='Foot'),
    yaxis=dict(title='Count'),
    hovermode='closest'
    )

    fig = go.Figure(data=data,layout=layout)
    fig.show()

    sent=['Neutral','Positive','Negative']

    for s in sent:
        x=parsed_and_scored_news[parsed_and_scored_news['Sentiment']==s]['headline']

        x = parsed_and_scored_news['headline']
        text = ' '.join(x).lower()

        plt.subplots(figsize = (8,8))

        wordcloud = WordCloud (
                            background_color = 'white',
                            width = 512,
                            height = 384
                                ).generate(' '.join(x))
        plt.imshow(wordcloud) # image show
        plt.axis('off') # to off the axis of x and y
        plt.title(s+' News',fontsize=40,color="Black")
        plt.show()

        print('\n\n')
