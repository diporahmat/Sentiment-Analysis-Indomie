import json, tweepy, re, string, nltk
from requests_oauthlib import OAuth1
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
import matplotlib.pyplot as plt

with open("token.json")as f:
    tokens = json.load(f)

bearer_token = tokens['bearer_token']
api_key = tokens['api_key']
api_key_secret = tokens['api_key_secret']
access_token = tokens['access_token']
access_token_secret = tokens['access_token_secret']

auth = OAuth1(api_key,
             api_key_secret,
             access_token,
             access_token_secret
             )
api = tweepy.Client(bearer_token=bearer_token, wait_on_rate_limit=True)

# Data Collection
auth = OAuth1(api_key,
             api_key_secret,
             access_token,
             access_token_secret
             )
api = tweepy.Client(bearer_token=bearer_token, wait_on_rate_limit=True)

start_time = "2022-01-23T00:00:00+07:00"
end_time = "2022-01-26T00:00:00+07:00"
new_search = "indomie -is:retweet --lang:id"

tweets = tweepy.Paginator(api.search_recent_tweets,
        query=new_search,
        start_time=start_time,
        end_time=end_time).flatten(limit=4000)

items = []
for tweet in tweets:
    items.append (' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet.text).split()))
df = pd.DataFrame(data=items, columns=['tweet'])
items

# Save to csv
df.to_csv('tweets_indomie.csv')
df

# Data Preparation
df['clean_tweet'] = df['tweet'].str.lower()
df['clean_tweet'] = df['clean_tweet'].str.replace(r'\d+', '', regex=True)
df['clean_tweet'] = df['clean_tweet'].str.translate(str.maketrans("","",string.punctuation))
df['clean_tweet'] = df['clean_tweet'].str.strip()
df['clean_tweet'] = df['clean_tweet'].str.split()
df

# Remove Emoji
def emoji_cleaner(data):
    word_emoji = data
    word_list = []
    
    emoji_pattern = re.compile("["
                u"\U0001F600-\U0001F64F"  # emoticons
                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                    "]+", flags=re.UNICODE)
    
    for word in word_emoji:
        emoji_pattern.sub(r'', word) 
        word_list.append(word)
    data = ' '.join(word_list)
    return data

df['clean_tweet'] = df['clean_tweet'].apply(lambda x: emoji_cleaner(x))
df

# Remove Slank Word
slang_list = [('yg','yang'),
              ('ga','tidak'),
              ('gak','tidak'),
              ('ya','iya'),
              ('aja','saja'),
              ('kalo','kalau'),
              ('gue','saya'),
              ('aku','saya'),
              ('bgt','banget')
             ]
def tokenizing(data):
    data = word_tokenize(data)
    return data

def slang_cleaner(data, slang_list:list=[]):
    word_token = tokenizing(data)
    word_list = []
    
    for word in word_token:
        for slang in slang_list:
            if word.lower() in slang[0].lower():
                word = slang[1]
                break
        word_list.append(word)
        
    data = ' '.join(word_list)
    return data

df['clean_tweet'] = df['clean_tweet'].apply(lambda x: slang_cleaner(x, slang_list))
df

# Remove Stopwords
df['clean_tweet'] = df['clean_tweet'].str.strip()

df['clean_tweet'] = df['clean_tweet'].str.split()
def clean_stopwords(word_list):
    processed_word_list = []
    for word in word_list:
        word = word.lower() # in case they arenet all lower cased
        if word not in stopwords.words("Indonesian"):
            processed_word_list.append(word)
    return processed_word_list
df['clean_tweet'] = df['clean_tweet'].apply(clean_stopwords)
df

# Join Words
df['clean_tweet'] = df['clean_tweet'].str.join(' ')
df

# Data Processing
pos_list= open("./kata_positif.txt","r")
pos_kata = pos_list.readlines()
neg_list= open("./kata_negatif.txt","r")
neg_kata = neg_list.readlines()

hasil = []
list_anti = ['tidak','lawan','anti', 'belum', 'belom', 'tdk', 'jangan', 'gak', 'enggak', 'bukan', 'sulit', 'tak', 'sblm']

for item in items:
    print(item.strip())
    tweets = item.strip().split() #tokenization
    
    count_p = 0 #nilai positif
    count_n = 0 #nilai negatif
    
    for tweet in tweets:
        for kata_pos in pos_kata:
            if kata_pos.strip().lower() == tweet.lower():
                if items[items.index(item)-1] in list_anti:
                    print(items[items.index(item)-1], kata_pos, ['negatif'])
                    count_n += 1
                else:
                    print(kata_pos, ['positif'])
                    count_p += 1
        for kata_neg in neg_kata:
            if kata_neg.strip().lower() == tweet.lower():
                if items[items.index(item)-1] in list_anti:
                    print(items[items.index(item)-1], kata_neg, ['positif'])
                    count_p += 1
                else:
                    print(kata_neg, ['negatif'])
                    count_n += 1
    
    hasil.append(count_p - count_n)
    
print ("Nilai rata-rata: "+str(np.mean(hasil)))
print ("Standar deviasi: "+str(np.std(hasil)))
df['sentiment'] = hasil
df

# Change number to word in sentiment
def change_sentiment(sentiments):
    sentiments = str(sentiments)
    processed_sentiment = []
    for sentiment in sentiments:
        if sentiment == '-' :
            processed_sentiment.append('negative')
        else:
            processed_sentiment.append('positive')
    return processed_sentiment
df['sentiment'] = df['sentiment'].apply(change_sentiment)
df['sentiment'] = df['sentiment'].str.join(' ')
df

# Share Insight
#character count on tweets
bin_range = np.arange(0, 260, 10)
df['clean_tweet'].str.len().hist(bins=bin_range)
plt.show()

#word count on tweets
bin_range = np.arange(0, 50)
df['clean_tweet'].str.split().map(lambda x: len(x)).hist(bins=bin_range)
plt.show()

#Average word length on tweets
df['clean_tweet'].str.split().apply(lambda x : [len(i) for i in x]).map(lambda x: np.mean(x)).hist()
plt.show()

#word tokenizing
df['clean_tweet'] = df['clean_tweet'].apply(lambda x: word_tokenize(str(x)))
tweets = [word for tweet in df['clean_tweet'] for word in tweet]

#Frequency Distribution
fqdist = FreqDist(tweets)
print(fqdist)

#Most Common Word
most_common_word = fqdist.most_common(20)

print(most_common_word)

#Plot of Frequency Distribution
fqdist.plot(20,cumulative=False)
plt.show()

#Ngrams
result = pd.Series(nltk.ngrams(tweets, 2)).value_counts()[:20]
print (result)

#Visualization of Ngrams
labels, counts = np.unique(hasil, return_counts=True)
plt.bar(labels, counts, align='center')
plt.gca().set_xticks(labels)
plt.show()

# Data Analysis
'''From the data I got from Twitter, I collected 4000 tweets from January 23, 2022 to January 26, 2022. From this data, Indomie products received a positive response from general worthy things (here tweeters) and the word that was widely discussed was eating fried indomie. The indomie product that is widely discussed by the public is fried indomie.'''