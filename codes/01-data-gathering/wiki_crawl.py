# %% [markdown]
# # Wikipedia Crawling

# %% [markdown]
# ### Imports

# %%
import wikipedia
import nltk
import string 
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# %% [markdown]
# ### Set user parameters

# %%
# PARAMETERS 
label_list=["Women's rights", "Men's rights"]

max_num_pages=30
sentence_per_chunk=10
min_sentence_length=20

# GET STOPWORDS
from nltk.corpus import stopwords
stop_words=nltk.corpus.stopwords.words('english')

# INITALIZE STEMMER+LEMITZIZER+SIA
sia = SentimentIntensityAnalyzer()
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# %% [markdown]
# ### Define text cleaning function

# %%
def clean_string(text):
	# #FILTER OUT UNWANTED CHAR
	new_text=""
	# keep=string.printable
	keep=" abcdefghijklmnopqrstuvwxyz0123456789"
	for character in text:
		if character.lower() in keep:
			new_text+=character.lower()
		else: 
			new_text+=" "
	text=new_text
	# print(text)

	# #FILTER OUT UNWANTED WORDS
	new_text=""
	for word in nltk.tokenize.word_tokenize(text):
		if word not in stop_words:
			#lemmatize 
			tmp=lemmatizer.lemmatize(word)
			# tmp=stemmer.stem(tmp)

			# update word if there is a change
			# if(tmp!=word): print(tmp,word)
			
			word=tmp
			if len(word)>1:
				if word in [".",",","!","?",":",";"]:
					#remove the last space
					new_text=new_text[0:-1]+word+" "
				else: #add a space
					new_text+=word.lower()+" "
	text=new_text.strip()
	return text

# %% [markdown]
# ### Perform Wiki crawl

# %%
#INITIALIZE 
corpus=[]  # list of strings (input variables X)
targets=[] # list of targets (labels or response variables Y)

#--------------------------
# LOOP OVER TOPICS 
#--------------------------
for label in label_list:

	#SEARCH FOR RELEVANT PAGES 
	titles=wikipedia.search(label,results=max_num_pages)
	print("Pages for label =",label,":",titles)

	#LOOP OVER WIKI-PAGES
	for title in titles:
		try:
			print("	",title)
			wiki_page = wikipedia.page(title, auto_suggest=True)

			# LOOP OVER SECTIONS IN ARTICLE AND GET PAGE TEXT
			for section in wiki_page.sections:
				text=wiki_page.section(section); #print(text)

				#BREAK IN TO SENTENCES 
				sentences=nltk.tokenize.sent_tokenize(text)
				counter=0
				text_chunk=''

				#LOOP OVER SENTENCES 
				for sentence in sentences:
					if len(sentence)>min_sentence_length:
						if(counter%sentence_per_chunk==0 and counter!=0):
							# PROCESS COMPLETED CHUNK 
							
							# CLEAN STRING
							text_chunk=clean_string(text_chunk)

							# REMOVE LABEL IF IN STRING (MAKES IT TOO EASY)
							text_chunk=text_chunk.replace(label,"")
							
							# REMOVE ANY DOUBLE SPACES
							text_chunk=' '.join(text_chunk.split()).strip()

							#UPDATE CORPUS 
							corpus.append(text_chunk)

							#UPDATE TARGETS
							score=sia.polarity_scores(text_chunk)
							target=[label,score['compound']]
							targets.append(target)

							#print("TEXT\n",text_chunk,target)

							# RESET CHUNK FOR NEXT ITERATION 
							text_chunk=sentence
						else:
							text_chunk+=sentence
						#print("--------\n", sentence)
						counter+=1

		except:
			print("WARNING: SOMETHING WENT WRONG:", title);  


# %% [markdown]
# ### Save results

# %%
#SANITY CHECKS AND PRINT TO FILE 
print("number of text chunks = ",len(corpus))
print("number of targets = ",len(targets))

tmp=[]
for i in range(0,len(corpus)):
    tmp.append([corpus[i],targets[i][0],targets[i][1]])
df=pd.DataFrame(tmp)
df=df.rename(columns={0: "text", 1: "label", 2: "sentiment"})
print(df)
df.to_csv('wiki-crawl-results.csv',index=False)


