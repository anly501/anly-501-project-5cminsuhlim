import numpy as np
import pandas as pd
import os
import time
import requests
import json
import csv
from tqdm import tqdm
import tweepy
from datetime import datetime, timedelta


api = pd.read_csv("./twitterapiauth.txt", sep=" ", header=None)

consumer_key        = api.loc[0,1]
consumer_secret     = api.loc[1,1]
access_token        = api.loc[2,1]
access_token_secret = api.loc[3,1]
bearer_token        = api.loc[4,1]

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API()
headers = {"Authorization": "Bearer {}".format(bearer_token)}


# search_twitter function
def search_twitter(max_results, query, tweet_fields, start_time, end_time, bearer_token):
    url = "https://api.twitter.com/2/tweets/search/recent?max_results={}&query={}&start_time={}&end_time={}&{}".format(
        max_results, query, start_time, end_time, tweet_fields
    )
    
    response = requests.request("GET", url, headers=headers)

    print(response.status_code)

    if response.status_code != 200:
        raise Exception(response.status_code, response.text)
    return response.json()

# pretty print function
def pretty_print_json(input):
    print(json.dumps(input, indent=4, sort_keys=True))
    

max_results = 100
query = "%23wagegap%20OR%20%23earningsgap%20OR%20%23feminism%20OR%20%23men%27srights%20OR%20%23women%27srights%20OR%20%23MGTOW%20"
tweet_fields = "tweet.fields=text,lang"

# gather the 100 tweets each over the last week. x CANNOT exceed 6
def collect(x=6):
    l = []
    dtformat = '%Y-%m-%dT%H:%M:%SZ'
    time = datetime.utcnow()
    for i in range(1, x + 1):
        start_time = time - timedelta(days=i + 1)
        end_time = time - timedelta(days=i)
        start_time, end_time = start_time.strftime(dtformat), end_time.strftime(dtformat)
        json_response = search_twitter(max_results=max_results, query=query, tweet_fields=tweet_fields, start_time=start_time, end_time=end_time, bearer_token=bearer_token)
        dictJson = json.loads(json.dumps(json_response))
        l.append(dictJson)
    return l


dictList = collect()

tweets_list = []

for tweets in dictList:
    for tweet in tweets['data']:
        if tweet['lang'] == 'en':
            tweets_list.append(tweet['text'].split())

words = [inner for outer in tweets_list for inner in outer]
d = {}

for word in words:
    d[word] = d.get(word, 0) + 1

df = pd.DataFrame(d.items(), columns=['Word', 'Count'])
pd.DataFrame.to_csv(df, "Tweets.csv")