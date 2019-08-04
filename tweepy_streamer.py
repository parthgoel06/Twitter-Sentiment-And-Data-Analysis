from tweepy import API
from tweepy import OAuthHandler
from textblob import TextBlob
import twitter_credentials
import re
import os
import itertools
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update(
    {
        'text.usetex': False,
        'font.family': 'stixgeneral',
        'mathtext.fontset': 'stix',
    }
)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


class TwitterClient:

    def __init__(self, twitter_user=None):
        self.auth = TwitterAuthenticator().authenticate_twitter_app()
        self.twitter_client = API(self.auth)
        self.twitter_user = twitter_user

    def get_twitter_client_api(self):
        return self.twitter_client


class TwitterAuthenticator:

    def authenticate_twitter_app(self):
        auth = OAuthHandler(twitter_credentials.CONSUMER_KEY, twitter_credentials.CONSUMER_SECRET)
        auth.set_access_token(twitter_credentials.ACCESS_TOKEN, twitter_credentials.ACCESS_TOKEN_SECRET)
        return auth


class TweetAnalyzer:

    def clean_tweet(self, tweet):
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

    def analyze_sentiment(self, tweet):
        analysis = TextBlob(self.clean_tweet(tweet))
        if analysis.sentiment.polarity > 0:
            return 1
        elif analysis.sentiment.polarity == 0:
            return 0
        else:
            return -1

    def tweets_to_data_frame(self, tweets):
        rdf = pd.DataFrame(data=[tweet.text for tweet in tweets], columns=['Tweets'])
        rdf['ID'] = np.array([tweet.id for tweet in tweets])
        rdf['Length'] = np.array([len(tweet.text) for tweet in tweets])
        rdf['Date'] = np.array([tweet.created_at for tweet in tweets])
        rdf['Source'] = np.array([tweet.source for tweet in tweets])
        rdf['Likes'] = np.array([tweet.favorite_count for tweet in tweets])
        rdf['Retweets'] = np.array([tweet.retweet_count for tweet in tweets])
        rdf['User'] = np.array([tweet.user.screen_name for tweet in tweets])
        return rdf

    def tweets_to_data_frame2(self, tweets2):
        adf = pd.DataFrame(data=[tweet.text for tweet in tweets2], columns=['Tweets'])
        adf['ID'] = np.array([tweet.id for tweet in tweets2])
        adf['Length'] = np.array([len(tweet.text) for tweet in tweets2])
        adf['Date'] = np.array([tweet.created_at for tweet in tweets2])
        adf['Source'] = np.array([tweet.source for tweet in tweets2])
        adf['Likes'] = np.array([tweet.favorite_count for tweet in tweets2])
        adf['Retweets'] = np.array([tweet.retweet_count for tweet in tweets2])
        adf['User'] = np.array([tweet.user.screen_name for tweet in tweets2])
        return adf


if __name__ == '__main__':
    twitter_client = TwitterClient()
    tweet_analyzer = TweetAnalyzer()
    api = twitter_client.get_twitter_client_api()

    tweets = api.search(q="@Reliancejio", count=100, lang="en")
    tweets2 = api.search(q="@airtelindia", count=100, lang="en")

    rdf = tweet_analyzer.tweets_to_data_frame(tweets)
    adf = tweet_analyzer.tweets_to_data_frame2(tweets2)

    rdf['Sentiment'] = np.array([tweet_analyzer.analyze_sentiment(tweet) for tweet in rdf['Tweets']])
    adf['Sentiment'] = np.array([tweet_analyzer.analyze_sentiment(tweet) for tweet in adf['Tweets']])

    date_format = "%Y-%m-%dT%H:%M:%S"
    rdf["Date"] = pd.to_datetime(rdf["Date"], format=date_format)
    adf["Date"] = pd.to_datetime(adf["Date"], format=date_format)
    rdf["Hour"] = pd.DatetimeIndex(rdf["Date"]).hour
    adf["Hour"] = pd.DatetimeIndex(adf["Date"]).hour
    adf["Month"] = pd.DatetimeIndex(adf["Date"]).month
    rdf["Month"] = pd.DatetimeIndex(rdf["Date"]).month
    rdf["Day"] = pd.DatetimeIndex(rdf["Date"]).day
    adf["Day"] = pd.DatetimeIndex(adf["Date"]).day
    rdf["Month_f"] = rdf["Month"].map({1: "JAN", 2: "FEB", 3: "MAR", 4: "APR", 5: "MAY", 6: "JUN", 7: "JUL",
                                       8: "AUG", 9: "SEP"})
    adf["Month_f"] = adf["Month"].map({1: "JAN", 2: "FEB", 3: "MAR", 4: "APR", 5: "MAY", 6: "JUN", 7: "JUL",
                                       8: "AUG", 9: "SEP"})

    print("\nRELIANCE JIO TWEETS\n")
    print(rdf)
    print("\nAIRTEL INDIA TWEETS\n")
    print(adf)

    exists = os.path.isfile(r'RelianceJio_DB.csv')
    exists2 = os.path.isfile(r'Airtel_DB.csv')

    if exists:
        rdf.to_csv(r'RelianceJio_DB.csv', mode='a', header=False)
    else:
        rdf.to_csv(r'RelianceJio_DB.csv', mode='a')
    if exists2:
        adf.to_csv(r'Airtel_DB.csv', mode='a', header=False)
    else:
        adf.to_csv(r'Airtel_DB.csv', mode='a')

    rdf2 = pd.read_csv(r"RelianceJio_DB.csv")
    adf2 = pd.read_csv(r"Airtel_DB.csv")

    rdf2 = rdf2.drop_duplicates(subset='Tweets')
    adf2 = adf2.drop_duplicates(subset='Tweets')

    rdf2.to_csv(r'RelianceJio_DB.csv', index=False, mode='w')
    adf2.to_csv(r'Airtel_DB.csv', index=False, mode='w')

    rdf3 = rdf2.groupby('Source')['Sentiment'].sum().reset_index()
    adf3 = adf2.groupby('Source')['Sentiment'].sum().reset_index()
    ardf3 = pd.merge(rdf3, adf3, on="Source", suffixes=('_Reliance', '_Airtel'))

    print("\nSource-Sentiment Analysis\n")
    print(ardf3)

    rdf4 = rdf2.loc[(rdf2['Likes'] >= 1) & (rdf2['Sentiment'] < 0)]
    adf4 = adf2.loc[(adf2['Likes'] >= 1) & (adf2['Sentiment'] < 0)]

    rdf5 = rdf2.loc[(rdf2['Retweets'] >= 1) & (rdf2['Sentiment'] < 0)]
    adf5 = adf2.loc[(rdf2['Retweets'] >= 1) & (adf2['Sentiment'] < 0)]

    print('\nLikes and Retweets Analysis')
    print('\nNegative tweets got ' + str(rdf4['Likes'].sum()) + ' likes for Reliance Jio')
    print('Negative tweets got ' + str(adf4['Likes'].sum()) + ' likes for Airtel India')

    print('Negative tweets got ' + str(rdf5['Retweets'].sum()) + ' retweets for Reliance Jio')
    print('Negative tweets got ' + str(adf5['Retweets'].sum()) + ' retweets for Airtel India')

# Source-Sentiment Graph
    ax = ardf3.plot(x='Source', y='Sentiment_Reliance')
    ardf3.plot(ax=ax, x='Source', y='Sentiment_Airtel', figsize=(10, 4))
    ax.legend(["Reliance Jio Sentiment", "Airtel India Sentiment"])
    ax.relim()
    ax.autoscale_view()
    tick_labels = tuple(ardf3['Source'])
    x_max = int(max(plt.xticks()[0]))
    plt.xticks(range(0, x_max + 1), tick_labels, rotation=45)
    plt.ylabel("Cumulative Sentiment")
    plt.title("Source-Sentiment Graph")
    fig = plt.gcf()
    plt.show()
    fig.savefig(r'graphs\1.png', bbox_inches='tight')

# Sentiment Distribution Graph
    a1 = rdf2[rdf2['Sentiment'] == -1].shape[0]
    a2 = adf2[adf2['Sentiment'] == -1].shape[0]
    b1 = rdf2[rdf2['Sentiment'] == 0].shape[0]
    b2 = adf2[adf2['Sentiment'] == 0].shape[0]
    c1 = rdf2[rdf2['Sentiment'] == 1].shape[0]
    c2 = adf2[adf2['Sentiment'] == 1].shape[0]
    data = {'Reliance Jio': {'-1': a1, '0': b1, '1': c1}, 'Airtel India': {'-1': a2, '0': b2, '1': c2}}
    df = pd.DataFrame(data)
    df.plot(kind='bar', figsize=(10, 4))
    plt.xlabel("Sentiment")
    plt.ylabel("Tweets Count")
    plt.title("Sentiment Distribution Graph")
    fig = plt.gcf()
    plt.show()
    fig.savefig(r'graphs\2.png', bbox_inches='tight')

# Sentiment of tweets by hour of day
    st_hr_r = pd.crosstab(rdf2["Hour"], rdf2["Sentiment"])
    st_hr_r = st_hr_r.apply(lambda r: r/r.sum()*100, axis=1)

    st_hr_a = pd.crosstab(adf2["Hour"], adf2["Sentiment"])
    st_hr_a = st_hr_a.apply(lambda r: r/r.sum()*100, axis=1)

    st_hr_r.plot(kind="bar", figsize=(14, 7), color=["r", "b", "g"], linewidth=1, edgecolor="w", stacked=True)
    plt.legend(loc="best", prop={"size": 13})
    plt.title("Sentiment of tweets by hour of day - Reliance Jio")
    plt.xticks(rotation=0)
    plt.ylabel("Percentage")
    fig = plt.gcf()
    plt.show()
    fig.savefig(r'graphs\3a.png', bbox_inches='tight')

    st_hr_a.plot(kind="bar", figsize=(14, 7), color=["r", "b", "g"], linewidth=1, edgecolor="w", stacked=True)
    plt.legend(loc="best", prop={"size": 13})
    plt.title("Sentiment of tweets by hour of day - Airtel India")
    plt.xticks(rotation=0)
    plt.ylabel("Percentage")
    fig = plt.gcf()
    plt.show()
    fig.savefig(r'graphs\3b.png', bbox_inches='tight')


# Average retweets and likes by sentiment
    avg_lk_rts_r = rdf2.groupby("Sentiment")[["Retweets", "Likes"]].mean()
    avg_lk_rts_a = adf2.groupby("Sentiment")[["Retweets", "Likes"]].mean()

    avg_lk_rts_r.plot(kind="bar", figsize=(12, 6), linewidth=1, edgecolor="k")
    plt.xticks(rotation=0)
    plt.ylabel("Average")
    plt.title("Average retweets and likes by sentiment - Reliance Jio")
    fig = plt.gcf()
    plt.show()
    fig.savefig(r'graphs\4a.png', bbox_inches='tight')

    avg_lk_rts_a.plot(kind="bar", figsize=(12, 6), linewidth=1, edgecolor="k")
    plt.xticks(rotation=0)
    plt.ylabel("Average")
    plt.title("Average retweets and likes by sentiment - Airtel India")
    fig = plt.gcf()
    plt.show()
    fig.savefig(r'graphs\4b.png', bbox_inches='tight')

# Likes and retweets by sentiment

    lst = [-1, 0, 1]
    cs = ["r", "g", "b"]

    plt.figure(figsize=(13, 13))

    for i, j, k in itertools.zip_longest(lst, range(len(lst)), cs):
        plt.subplot(2, 2, j+1)
        plt.scatter(x=rdf2[rdf2["Sentiment"] == i]["Likes"], y=rdf2[rdf2["Sentiment"] == i]["Retweets"],
                    label="Reliance Jio", linewidth=.7, edgecolor="w", s=60, alpha=.7)
        plt.scatter(x=adf2[adf2["Sentiment"] == i]["Likes"], y=adf2[adf2["Sentiment"] == i]["Retweets"],
                    label="Airtel India", linewidth=.7, edgecolor="w", s=60, alpha=.7)
        plt.title(str(i) + " - Tweets")
        plt.legend(loc="best", prop={"size": 12})
        plt.xlabel("Like count")
        plt.ylabel("Retweet count")

        fig = plt.gcf()
        plt.show()
        fig.savefig(r'graphs\5_{0}.png'.format(i), bbox_inches='tight')

# Popular hashtags
    hashs_r = rdf2["Tweets"].str.extractall(r'(\#\w+)')[0].value_counts().reset_index()
    hashs_r.columns = ["Hashtags", "Count"]
    hashs_a = adf2["Tweets"].str.extractall(r'(\#\w+)')[0].value_counts().reset_index()
    hashs_a.columns = ["Hashtags", "Count"]
    plt.figure(figsize=(10, 20))
    plt.subplot(211)
    ax = sns.barplot(x="Count", y="Hashtags", data=hashs_r[:25], palette="seismic", linewidth=1, edgecolor="k" * 25)
    plt.grid(True)
    for i, j in enumerate(hashs_r["Count"][:25].values):
        ax.text(3, i, j, fontsize=10, color="white")
    plt.title("Popular hashtags used for Reliance Jio")
    plt.subplot(212)
    ax1 = sns.barplot(x="Count", y="Hashtags", data=hashs_a[:25], palette="seismic", linewidth=1, edgecolor="k" * 25)
    plt.grid(True)
    for i, j in enumerate(hashs_a["Count"][:25].values):
        ax1.text(.3, i, j, fontsize=10, color="white")
    plt.title("Popular hashtags used for Airtel India")
    fig = plt.gcf()
    plt.show()
    fig.savefig(r'graphs\6.png', bbox_inches='tight')

# Popular words in tweets
    pop_words_r = (rdf2["Tweets"].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index().sort_values(by=[0], ascending=False))
    pop_words_r.columns = ["Word", "Count"]
    pop_words_r["Word"] = pop_words_r["Word"].str.upper()

    pop_words_a = (adf2["Tweets"].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index().sort_values(by=[0], ascending=False))
    pop_words_a.columns = ["Word", "Count"]
    pop_words_a["Word"] = pop_words_a["Word"].str.upper()

    plt.figure(figsize=(12, 25))
    plt.subplot(211)
    ax = sns.barplot(x="Count", y="Word", data=pop_words_r[:30], linewidth=1, edgecolor="k" * 30, palette="Reds")
    plt.title("popular words in tweets - Reliance Jio")
    plt.grid(True)
    for i, j in enumerate(pop_words_r["Count"][:30].astype(int)):
        ax.text(.8, i, j, fontsize=9)
    plt.subplot(212)
    ax1 = sns.barplot(x="Count", y="Word", data=pop_words_a[:30], linewidth=1, edgecolor="k" * 30, palette="Blues")
    plt.title("Popular words in tweets - Airtel India")
    plt.grid(True)
    for i, j in enumerate(pop_words_a["Count"][:30].astype(int)):
        ax1.text(.8, i, j, fontsize=9)
    fig = plt.gcf()
    plt.show()
    fig.savefig(r'graphs\7.png', bbox_inches='tight')
