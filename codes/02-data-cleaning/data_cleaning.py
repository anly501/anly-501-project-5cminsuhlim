# %% [markdown]
# # Tweets

# %%
import pandas as pd
import numpy as np
import re
import string
import nltk
from sklearn.feature_extraction.text import CountVectorizer

# %%
raw_tweets = pd.read_json('../01-data-gathering/tweets.json')

# %%
n = len(raw_tweets.keys()) - 1
tweets_l = []
for i in range(0, n):
    if(raw_tweets[str(i)]['lang'] == 'en'):
        tweets_l.append(raw_tweets[str(i)]['text'])
tweets_l = list(dict.fromkeys(tweets_l))

# %%
#FILTER OUT UNWANTED CHAR
tweets_printable = []

for text in tweets_l:
    new_text=""
    for character in text:
        if character in string.printable:
            new_text+=character
    tweets_printable.append(new_text)

# %%
initial_clean = []

for tweet in tweets_printable:
    clean = re.sub(r"@[A-Za-z0-9_]+", "", tweet)
    clean = re.sub(r'http\S+', "", clean)
    clean = re.sub(r'https\S+', "", clean)
    clean = re.sub(r'www\S+', "", clean)
    clean = clean.strip()
    initial_clean.append(clean)

# %%
tweets_hashtags = []
for tweet in initial_clean:
    hashtags = re.findall("#([a-zA-Z0-9_]{1,50})", tweet)
    if hashtags:
        tweets_hashtags.append(hashtags)

# %%
flat_list = [hashtag for sublist in tweets_hashtags for hashtag in sublist]

# %%
stop = nltk.corpus.stopwords.words('english')

tokenized = [nltk.tokenize.word_tokenize(tweet.lower().strip()) for tweet in flat_list]
no_stopwords = []

for tweet in tokenized:
    for word in tweet:
        if word not in stop:
            no_stopwords.append(word)

# ref: https://stackoverflow.com/questions/10017147/removing-a-list-of-characters-in-string
to_remove = [".",",","!","?",":",";","_"]
cleaned_tweets = [tweet.translate({ord(x): '' for x in to_remove}) for tweet in no_stopwords]

# %%
vectorizer=CountVectorizer()   

Xs  =  vectorizer.fit_transform(cleaned_tweets)   
tweets = pd.DataFrame.from_dict(vectorizer.vocabulary_, orient='index')
tweets.reset_index(inplace=True)
tweets.columns = ['Word', 'Count']

# %%
tweets.replace('', np.nan, inplace=True)
tweets.dropna(inplace=True)
tweets.sort_values('Count', ascending=False, inplace=True)

# %%
tweets.to_csv('./../../data/01-modified-data/Tweets_final.csv')


