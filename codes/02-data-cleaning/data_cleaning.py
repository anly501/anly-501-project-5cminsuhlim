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
raw_tweets = pd.read_json('../../data/00-raw-data/tweets.json')

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
tweets.reset_index(inplace=True)

# remove incomplete / nonsense words
to_drop = [21, 23, 32, 40, 43, 48, 71, 119, 171, 178 , 208, 213, 217, 257, 260]
tweets.drop(to_drop,axis=0, inplace=True)

# %%
tweets.to_csv('./../../data/01-modified-data/Tweets_final.csv')

# %% [markdown]
# # Employment and Wages (BLS)

# %%
df = pd.read_excel('../../data/00-raw-data/wages_(by_occupation_may_2021).xlsx')

# %%
df = df[['OCC_TITLE', 'O_GROUP', 'TOT_EMP', 'EMP_PRSE', 'A_MEAN', 'MEAN_PRSE']]
df = df.iloc[1:, :]
df['Target'] = 'X'
df['Target_Num'] = 0
majors = df[df['O_GROUP'] == 'major']['OCC_TITLE']

# %%
# create string and numeric representations for major occupation titles
df.loc[1:73, 'Target'] = majors[1]
df.loc[1:73, 'Target_Num'] = 1
df.loc[74:131, 'Target'] = majors[74]
df.loc[74:131, 'Target_Num'] = 2
df.loc[132:167, 'Target'] = majors[132]
df.loc[132:167, 'Target_Num'] = 3
df.loc[168:228, 'Target'] = majors[168]
df.loc[168:228, 'Target_Num'] = 4
df.loc[229:307, 'Target'] = majors[229]
df.loc[229:307, 'Target_Num'] = 5
df.loc[308:333, 'Target'] = majors[308]
df.loc[308:333, 'Target_Num'] = 6
df.loc[334:348, 'Target'] = majors[334]
df.loc[334:348, 'Target_Num'] = 7
df.loc[349:445, 'Target'] = majors[349]
df.loc[349:445, 'Target_Num'] = 8
df.loc[446:507, 'Target'] = majors[446]
df.loc[446:507, 'Target_Num'] = 9
df.loc[508:609, 'Target'] = majors[508]
df.loc[508:609, 'Target_Num'] = 10
df.loc[610:636, 'Target'] = majors[610]
df.loc[610:636, 'Target_Num'] = 11
df.loc[637:679, 'Target'] = majors[637]
df.loc[637:679, 'Target_Num'] = 12
df.loc[680:712, 'Target'] = majors[680]
df.loc[680:712, 'Target_Num'] = 13
df.loc[713:730, 'Target'] = majors[713]
df.loc[713:730, 'Target_Num'] = 14
df.loc[731:790, 'Target'] = majors[731]
df.loc[731:790, 'Target_Num'] = 15
df.loc[791:833, 'Target'] = majors[791]
df.loc[791:833, 'Target_Num'] = 16
df.loc[834:942, 'Target'] = majors[834]
df.loc[834:942, 'Target_Num'] = 17
df.loc[943:966, 'Target'] = majors[943]
df.loc[943:966, 'Target_Num'] = 18
df.loc[967:1069, 'Target'] = majors[967]
df.loc[967:1069, 'Target_Num'] = 19
df.loc[1070:1144, 'Target'] = majors[1070]
df.loc[1070:1144, 'Target_Num'] = 20
df.loc[1145:1311, 'Target'] = majors[1145]
df.loc[1145:1311, 'Target_Num'] = 21
df.loc[1312:1402, 'Target'] = majors[1312]
df.loc[1312:1402, 'Target_Num'] = 22

# %%
# remove rows containing '*'
df = df[df.apply(lambda x: (~x.astype(str).str.contains('\*', case=True, regex=True)))].dropna()

# %%
df.to_csv('./../../data/01-modified-data/occupations_detailed_(employment_and_wage).csv')


