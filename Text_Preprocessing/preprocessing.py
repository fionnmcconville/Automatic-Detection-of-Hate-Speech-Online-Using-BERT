
import pandas as pd
import numpy as np
import html
import re
import json
import string
import nltk
import subprocess
import sys


#pip install within script requires below function
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install("emoji")
install("wordsegment")
import emoji
import wordsegment as ws
from wordsegment import load, segment
load()

SEED = 3060

def loadData(data1, data2 = None, options = None, dataset = None):

  if data2 is not None:
    frames = [data1,data2]
    data = pd.concat(frames)
  else:
    data = data1
    
  HASHTAG_SEGMENTATION = options[0]
  EMOJI_REPLACEMENT = options[1]
  LEMMATIZE = options[2]
  REMOVE_STOPWORDS = options[3]
  REMOVE_PUNCTUATION = options[4]
  
  #Replace emoji must be done before basic preprocess otherwise unicode will be wiped out
  #And this function will be ineffective
  if EMOJI_REPLACEMENT == 'Replace_Emoji_v1':
    data['tweet'] = data['tweet'].apply(emojiReplace)

  if EMOJI_REPLACEMENT == 'Replace_Emoji_v2':
    data['tweet'] = data['tweet'].apply(emojiReplace_v2)

  #Must be performed after emoji translation
  data['tweet'] = data['tweet'].apply(preprocess)

  if HASHTAG_SEGMENTATION == True:    
    data['tweet'] = data['tweet'].apply(hashtagSegment)

  if REMOVE_PUNCTUATION == True:
    data['tweet'] = data['tweet'].apply(lambda x: remove_punct(x))

  if REMOVE_STOPWORDS == True:
    nltk.download('stopwords')
    data['tweet'] = data['tweet'].apply(lambda x: remove_stopwords(x))

  if LEMMATIZE == True:
    nltk.download('wordnet')
    data['tweet'] = data['tweet'].apply(lambda x: lemmatizing(x))

  #Remove small sequences that could skew model
  #data = data[data['tweet'].apply(lambda x: len(x) > 10)]
  data.dropna(inplace = True)
  data.reset_index(drop = True, inplace = True) 
  if dataset == "AnalyticsVidhya"  and len(data.index) < 20000:
    return data
  else:
    #We don't shuffle data when it is the analytics vidhya test set
    data = data.sample(frac = 1, random_state=SEED) # Shuffle data 
  return data

def preprocess(text_string):
    """
    Accepts a text string and:
    1) Removes URLS
    2) lots of whitespace with one instance
    3) Replaces mentions with common tags
    4) Uses the html.unescape() method to convert unicode to text counterpart
    5) Replace & with and
    6) Remove the fact the tweet is a retweet if it is - (knowing the tweet is 
       a retweet does not help towards our classification task).
    """
    space_pattern = '\s+'
    giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[#$-_@.&+]|'
        '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = '@[\w\-]+:'
    mention_regex1 = '@[\w\-]+'
    multmention_regex =  '(user ){2,}'
    #Sort out odd numbers of users
    multmention_regex1 = '(multuser user)'
    RT_regex = '(RT|rt)[ ]*@[ ]*[\S]+'
    
    # Replaces urls with URL
    parsed_text = re.sub(giant_url_regex, ' ', text_string)
    parsed_text = re.sub('URL', ' ', parsed_text)
    
    # Remove the fact the tweet is a retweet. 
    # (we're only interested in the language of the tweet here)
    parsed_text = re.sub(RT_regex, ' ', parsed_text) 
    
    # Removes mentions as they're redundant information
    parsed_text = re.sub(mention_regex, 'user',  parsed_text)
    #...including mentions with colons after - this seems to come up often
    parsed_text = re.sub(mention_regex1, 'user',  parsed_text)  

    #For multiple users
    parsed_text = re.sub(multmention_regex, 'multuser ',  parsed_text)
    parsed_text = re.sub(multmention_regex1, 'multuser ',  parsed_text)

    #Replace &amp; with and
    parsed_text = re.sub('&amp;', ' and', parsed_text)
    parsed_text = re.sub('&', ' and', parsed_text)

    # Remove unicode
    parsed_text = re.sub(r'[^\x00-\x7F]',' ', parsed_text) 
    parsed_text = re.sub(r'&#[0-9]+;', ' ', parsed_text)  

    # Convert unicode missed by above regex to text
    parsed_text = html.unescape(parsed_text)
    
    # Remove excess whitespace at the end
    parsed_text = re.sub(space_pattern, ' ', parsed_text) 
    
    # Set text to lowercase and strip
    parsed_text = parsed_text.lower()
    parsed_text = parsed_text.strip()
    
    return parsed_text

def emojiReplace(text_string):
    for word in text_string:
        if word in emoji.UNICODE_EMOJI:
            emoji_token = re.sub("[_-]", " ", emoji.demojize(word, delimiters = (" ", " "),  use_aliases = True))
            emoji_token = ' '.join(re.split('\W+', emoji_token)) + ' '
            text_string = text_string.replace(word, emoji_token)
            
            pattern = '(' + emoji_token + ')' + '{2,}'
            text_string = re.sub(pattern, 'multiple' + emoji_token + ' ', text_string)
    return re.sub("[-_]", " ", text_string)


def emojiReplace_v2(text_string):
    for word in text_string:
        if word in emoji.UNICODE_EMOJI:
            emoji_token = re.sub("[_-]", " ", emoji.demojize(word, delimiters = (" ", " ")))
            emoji_token = 'x'.join(re.split('\W+', emoji_token[1:])) + ' '
            text_string = text_string.replace(word, emoji_token)
            
            pattern = '(' + emoji_token + ')' + '{2,}'
            text_string = re.sub(pattern, 'mult' + emoji_token + ' ', text_string)
    return text_string


def hashtagSegment(text_string):

    """The values below of the bigrams reflect the amount of search results on google that come up"""
    # For example, we update wordsegment dict so it recognises altright as "alt right" rather than salt right
    ws.BIGRAMS['alt right'] = 1.17e8 

    ws.BIGRAMS['white supremacists'] = 3.86e6
    ws.BIGRAMS['tweets'] = 6.26e10
    ws.BIGRAMS['independece day'] = 6.21e7
    
    #Put a space before hashtags so each hashtag can be recognised separately 
    text_string = re.sub("#", " #", text_string)
    
    #We target hashtags so that we only segment the hashtag strings.
    #Otherwise the segment function may operate on misspelled words also; which
    #often appear in hate speech tweets owing to the ill education of those spewing it
    temp_str = []
    for word in text_string.split(' '):
        if word.startswith('#') == False:
            temp_str.append(word)
        else:
            temp_str = temp_str + segment(word)
            
    text_string = ' '.join(temp_str) 
    
    #Resolve excess whitespace
    text_string = re.sub('\s+', ' ', text_string)        
    return text_string

def remove_punct(text):
    #Return the charater as long as it's not punctuation
    text_nopunct = "".join([char for char in text if char not in string.punctuation])
    return text_nopunct

def lemmatizing(text):
    wn = nltk.WordNetLemmatizer()
    word_list = re.split('\W+', text)
    text = " ".join([wn.lemmatize(word) for word in word_list])
    return text


def remove_stopwords(text):
    stopwords = nltk.corpus.stopwords.words('english')
    word_list = re.split('\W+', text)
    text = " ".join([word for word in word_list if word not in stopwords])
    return text



