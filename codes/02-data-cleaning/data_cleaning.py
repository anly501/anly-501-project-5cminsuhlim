import pandas as pd
import re
from nltk.corpus import stopwords

tweets = pd.read_csv('../../data/00-raw-data/Tweets.csv')

stop = stopwords.words('english')
## remove leading and trailing whitespace
tweets['Word'] = tweets['Word'].str.strip()
## remove stopwords
tweets['Word'] = tweets['Word'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
## remove websites and non-alphanumeric characters
tweets['Word'] = tweets['Word'].replace(r'http\S+', '', regex=True).replace(r'www\S+', '', regex=True).replace(r'[^A-Za-z0-9]', '', regex=True)
## remove rows that are null or empty
tweets = tweets[(tweets['Word'].notnull()) & (tweets['Word']!='')]

tweets.to_csv('./../../data/01-modified-data/Tweets_final.csv')