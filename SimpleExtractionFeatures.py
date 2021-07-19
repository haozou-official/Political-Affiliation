# General
import pandas as pd
import numpy as np
import re
import jsonlines
import json
import glob
import os
from datetime import datetime

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.models.wrappers import LdaMallet

import warnings 
warnings.filterwarnings("ignore", category=DeprecationWarning)

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
# NLTK
import nltk
from nltk.corpus import stopwords
from nltk import bigrams

nltk.download('stopwords')
nltk.download('PorterStemmer')
nltk.download('punkt')

# User's ID
def getUserID(dataset):
    item = dataset['user']['id']
    try:
        if item is not None:
            return item
        else:
            raise TypeError
    
    except TypeError:
        print('None')

# user's name (firstname, last name)
def getUserName(dataset):
    item = dataset['user']['name']
    try:
        if item is not None:
            return item.lower()
        else:
            raise TypeError
    
    except TypeError:
        print('None')

# screen name
def getScreenName(dataset):
    item = dataset['user']['screen_name']
    try:
        if item is not None:
            return item.lower()
        else:
            raise TypeError
    
    except TypeError:
        print('None')
        
# length of user name -- screen name
def getUserNameLength(dataset):
    item = dataset['user']['screen_name']
    try:
        if item is not None:
            return len(item)
        else:
            raise TypeError
    
    except TypeError:
        print('None')
        
def getUserDescription(dataset):
    item = dataset['user']['description']
    try:
        if item is not None:
            return item.lower()
        else:
            raise TypeError
    
    except TypeError:
        print('None')
        
def getProfilePictureURL(dataset):
    item = dataset['user']['profile_image_url']
    try:
        if item is not None:
            return item
        else:
            raise TypeError
    
    except TypeError:
        print('None')
        
def getDateOfTweet(dataset):
    item = dataset['created_at']
    try:
        if item is not None:
            return item
        else:
            raise TypeError
    
    except TypeError:
        print('None')
        
def getDatetime(dataset):
    timestamp = int(dataset['timestamp_ms'])
    dt_object = datetime.fromtimestamp(timestamp / 1e3)
    return dt_object
        
def getNoLikes(dataset):
    item = dataset['user']['favourites_count']
    try:
        if item is not None:
            return item
        else:
            raise TypeError
    
    except TypeError:
        print('None')
        
def getNoFriends(dataset):
    item = dataset['user']['friends_count']
    try:
        if item is not None:
            return item
        else:
            raise TypeError
    
    except TypeError:
        print('None')

def getNoFollowers(dataset):
    item = dataset['user']['followers_count']
    try:
        if item is not None:
            return item
        else:
            raise TypeError
    
    except TypeError:
        print('None')
        
# def getNoTweets(dataset)
# tweet count = statuses_count
def getNoTweets(dataset):
    item = dataset['user']['statuses_count']
    try:
        if item is not None:
            return item
        else:
            raise TypeError
    
    except TypeError:
        print('None')
        
def getNoReplies(dataset):
    item = dataset['reply_count']
    try:
        if item is not None:
            return item
        else:
            raise TypeError
    
    except TypeError:
        print('None')

def getNoHashtags(dataset):
    item = dataset['entities']['hashtags']
    try:
        if item is not None:
            return len(item)
        else:
            raise TypeError
    
    except TypeError:
        print('None')

# info providers
def getNoURLs(dataset):
    item = dataset['entities']['urls']
    try:
        if item is not None:
            return len(item)
        else:
            raise TypeError
    
    except TypeError:
        print('None')
        
def getNoUserMentions(dataset):
    item = dataset['entities']['user_mentions']
    try:
        if item is not None:
            return len(item)
        else:
            raise TypeError
    
    except TypeError:
        print('None')
        
def getTweetLang(dataset):
    item = dataset['lang']
    try:
        if item is not None:
            return item
        else:
            raise TypeError
    
    except TypeError:
        print('None')

# truncation or not
def getTweetText(dataset):
    if 'retweeted_status' in dataset.keys():
        if dataset['retweeted_status']['truncated'] == True:
            text = "RT "+"@"+dataset['entities']['user_mentions'][0]['screen_name']+" "+dataset['retweeted_status']['extended_tweet']['full_text']
        else:
            text = "RT "+"@"+dataset['entities']['user_mentions'][0]['screen_name']+" "+dataset['retweeted_status']['text']
    else:
        text = dataset['text']
    return text

# helper func for getCleanTweet
def remove_users(tweet, pattern1, pattern2):
    r = re.findall(pattern1, tweet)
    for i in r:
        tweet = re.sub(i, '', tweet)
    
    r = re.findall(pattern2, tweet)
    for i in r:
        tweet = re.sub(i, '', tweet)
    return tweet

def remove_hashtags(tweet, pattern1, pattern2):
    r = re.findall(pattern1, tweet)
    for i in r:
        tweet = re.sub(i, '', tweet)
    
    r = re.findall(pattern2, tweet)
    for i in r:
        tweet = re.sub(i, '', tweet)
    return tweet

def remove_links(tweet):
    tweet_no_link = re.sub(r"http\S+", "", tweet)
    return tweet_no_link

# input: dataset
def getCleanTweet(dataset):
    tweet = getTweetText(dataset)
    # lower case
    tweet = tweet.lower()
    # remove twitter handles
    tidy_tweet = remove_users(tweet, "@ [\w]*", "@[\w]*")
    # remove tweet hashtags
    tidy_tweet = remove_hashtags(tidy_tweet, "# [\w]*", "#[\w]*")
    # remove hyperlinks
    tidy_tweet = remove_links(tidy_tweet)
    # Removing Punctuations, Numbers, and Special Characters
    tidy_tweet = tidy_tweet.replace("[^a-zA-Z#]", " ")
    # remove duplicate whitespace
    tidy_tweet = " ".join(tidy_tweet.split())
    # encode with utf-8
    tidy_tweet = tidy_tweet.encode()
    
    return tidy_tweet

# input: tweet
# Prepare Stop Words
stop_words = stopwords.words('english')
stop_words.extend(['from', 'https', 'twitter', 'religions', 'pic','twitt',])
def removeStopWords(tweets):
    return [[word for word in simple_preprocess(str(tweet)) if word not in stop_words] for tweet in tweets]

# input: tweet
ps = PorterStemmer()
def stemmingTweet(tweet):
    words = word_tokenize(tweet)
    stemm_lst = []
    for w in words:
        w = ps.stem(w)
        stemm_lst.append(w)
    return stemm_lst

# input: whole sentences (for analysis)
# extract bigram and trigram

# input: whole sentence_lst/tweet text list
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

def make_bigrams(sentences):
    data_words = list(sent_to_words(sentences))
    # Build the bigram and trigram model
    bigram = gensim.models.Phrases(data_words, min_count=10, threshold=100)
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)
    # Faster way to get a sentence clubbed as a bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    
    return [bigram_mod[doc] for doc in data_words]

def make_trigrams(sentences):
    data_words = list(sent_to_words(sentences))
    # Build the bigram and trigram model
    bigram = gensim.models.Phrases(data_words, min_count=10, threshold=100)
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)
    # Faster way to get a sentence clubbed as a bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    return [trigram_mod[bigram_mod[doc]] for doc in data_words]

def getBigrams(sentences):
    data_words = list(sent_to_words(sentences))
    return make_bigrams(data_words)

def getTrigrams(sentences):
    data_words = list(sent_to_words(sentences))
    return make_trigrams(data_words)

# Extract hashtags from the tweet
def getHashtags(dataset):
    item = dataset['entities']['hashtags']
    try:
        if item is not None:
            return item
        else:
            raise TypeError
    except TypeError:
        print('None')

# Extract mentions from the tweet
def getMentions(dataset):
    item = dataset['entities']['user_mentions']
    try:
        if item is not None:
            return item
        else:
            raise TypeError
    except TypeError:
        print('None')

# Extract hyperlinks
def getURLs(dataset):
    item = dataset['entities']['urls']
    try:
        if item is not None:
            return item
        else:
            raise TypeError
    
    except TypeError:
        print('None')

# Extract the origin if it is retweeted
def getOriginText(dataset):
    if 'retweeted_status' in dataset.keys():
        if dataset['retweeted_status']['truncated'] == True:
            text = dataset['retweeted_status']['extended_tweet']['full_text']
        else:
            text = dataset['retweeted_status']['text']
    else:
        text = dataset['text']
    return text

# one jsonl file
def createCSV(datasets, name):
    UID_lst = []
    Datetime_lst = []
    UserName_lst = []
    ScreenName_lst = []
    Description_lst = []
    URLs_lst = []
    TweetText_lst = []
    TweetLang_lst = []
    Trigrams_lst = []
    ProfilePictureURL_lst = []
    OriginalText_lst = []
    NoUserMentions_lst = []
    NoURLs_lst = []
    NoTweets_lst = []
    NoReplies_lst = []
    NoLikes_lst = []
    NoHashtags_lst = []
    NoFriends_lst = []
    NoFollowers_lst = []
    Mentions_lst = []
    Hashtags_lst = []
    CleanTweet_lst = []
    Bigrams_lst = []
    for i in range(len(datasets)):
        UID = getUserID(datasets[i])
        Datetime = getDatetime(datasets[i])
        UserName = getUserName(datasets[i])
        ScreenName = getScreenName(datasets[i])
        Description = getUserDescription(datasets[i])
        URLs = getURLs(datasets[i])
        TweetText = getTweetText(datasets[i])
        TweetLang = getTweetLang(datasets[i])
        Trigrams = getTrigrams(datasets[i])
        ProfilePictureURL = getProfilePictureURL(datasets[i])
        OriginalText = getOriginText(datasets[i])
        NoUserMentions = getNoUserMentions(datasets[i])
        NoURLs = getNoURLs(datasets[i])
        NoTweets = getNoTweets(datasets[i])
        NoReplies = getNoReplies(datasets[i])
        NoLikes = getNoLikes(datasets[i])
        NoHashtags = getNoHashtags(datasets[i])
        NoFriends = getNoFriends(datasets[i])
        NoFollowers = getNoFollowers(datasets[i])
        Mentions = getMentions(datasets[i])
        Hashtags = getHashtags(datasets[i])
        CleanTweet = getCleanTweet(datasets[i])
        Bigrams = getBigrams(datasets[i])
        
        UID_lst.append(UID)
        Datetime_lst.append(Datetime)
        UserName_lst.append(UserName)
        ScreenName_lst.append(ScreenName)
        Description_lst.append(Description)
        URLs_lst.append(URLs)
        TweetText_lst.append(TweetText)
        TweetLang_lst.append(TweetLang)
        Trigrams_lst.append(Trigrams)
        ProfilePictureURL_lst.append(ProfilePictureURL)
        OriginalText_lst.append(OriginalText)
        NoUserMentions_lst.append(NoUserMentions)
        NoURLs_lst.append(NoURLs)
        NoTweets_lst.append(NoTweets)
        NoReplies_lst.append(NoReplies)
        NoLikes_lst.append(NoLikes)
        NoHashtags_lst.append(NoHashtags)
        NoFriends_lst.append(NoFriends)
        NoFollowers_lst.append(NoFollowers)
        Mentions_lst.append(Mentions)
        Hashtags_lst.append(Hashtags)
        CleanTweet_lst.append(CleanTweet)
        Bigrams_lst.append(Bigrams)
        
    df = pd.DataFrame(UID_lst, columns=['UID'])
    df['Datetime'] = Datetime_lst
    df['User Name'] = UserName_lst
    df['Screen Name'] = ScreenName_lst
    df['Description'] = Description_lst
    df['URLs'] = URLs_lst
    df['Tweet Text'] = TweetText_lst
    df['Tweet Lang'] = TweetLang_lst
    df['Trigrams'] = Trigrams_lst
    df['Profile Picture URL'] = ProfilePictureURL_lst
    df['Original Text'] = OriginalText_lst
    df['No User Mentions'] = NoUserMentions_lst
    df['No URLs'] = NoURLs_lst
    df['No Tweets'] = NoTweets_lst
    df['No Replies'] = NoReplies_lst
    df['No Likes'] = NoLikes_lst
    df['No Hashtags'] = NoHashtags_lst
    df['No Friends'] = NoFriends_lst
    df['No Followers'] = NoFollowers_lst
    df['Mentions'] = Mentions_lst
    df['Hashtags'] = Hashtags_lst
    df['Clean Tweet'] = CleanTweet_lst
    df['Bigrams'] = Bigrams_lst
    

    df.to_csv('./us_presidential_election_2020/SimpleExtractionFeatures/'+name+'.csv', index=False)


def main():
    datasets = []
    filename_lst = []
    path = './us_presidential_election_2020/201006200213_vp_debate/'
    for filename in glob.glob(os.path.join(path, '*.json')):
        filename_lst.append(filename)
        
    for i in range(len(filename_lst)):
        datasets = []
        try:
            for line in open(filename_lst[i], 'r'):
                datasets.append(json.loads(line))
        except json.decoder.JSONDecodeError:
            print("json.decoder.JSONDecodeError: "+str(i))
        # eg. vp_debate
        name = 'vp_debate_vol_'+str(i)
        createCSV(datasets, name)


if __name__ == "__main__":
    main()
