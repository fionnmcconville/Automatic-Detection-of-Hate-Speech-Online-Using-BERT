
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


install("wordsegment")
install("demoji-0.1.5-py3-none-any.whl")
import wordsegment as ws
from wordsegment import load, segment
import demoji

demoji.download_codes()
nltk.download('wordnet')
nltk.download('stopwords')
load()

def preprocess(text_string):
    """
    Accepts a text string and:
    1) Removes URLS
    2) lots of whitespace with one instance
    3) Removes mentions
    4) Uses the html.unescape() method to convert unicode to text counterpart
    5) Replace & with and
    6) Remove the fact the tweet is a retweet if it is - knowing the tweet is 
       a retweet does not help towards our classification task.
    """
    space_pattern = '\s+'
    giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[#$-_@.&+]|'
        '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = '@[\w\-]+:'
    mention_regex1 = '@[\w\-]+'
    RT_regex = '(RT|rt)[ ]*@[ ]*[\S]+'
    
    # Replaces urls with URL
    parsed_text = re.sub(giant_url_regex, '', text_string)
    parsed_text = re.sub('URL', '', parsed_text)
    
    # Remove the fact the tweet is a retweet. 
    # (we're only interested in the language of the tweet here)
    parsed_text = re.sub(RT_regex, ' ', parsed_text) 
    
    # Removes mentions as they're redundant information
    parsed_text = re.sub(mention_regex, '',  parsed_text)
    #...including mentions with colons after - this seems to come up often
    parsed_text = re.sub(mention_regex1, '',  parsed_text)  

    #Replace &amp; with and
    parsed_text = re.sub('&amp;', 'and', parsed_text)

    # Remove unicode
    parsed_text = re.sub(r'[^\x00-\x7F]','', parsed_text) 
    parsed_text = re.sub(r'&#[0-9]+;', '', parsed_text)  

    # Convert unicode missed by above regex to text
    parsed_text = html.unescape(parsed_text)
    
    # Remove excess whitespace at the end
    parsed_text = re.sub(space_pattern, ' ', parsed_text) 
    
    # Set text to lowercase and strip
    parsed_text = parsed_text.lower()
    parsed_text = parsed_text.strip()
    
    return parsed_text


def emojiReplace(text_string):
    emoji_dict = demoji.findall(text_string)
    for emoji in emoji_dict.keys():
        text_string = text_string.replace(emoji, ' '+  emoji_dict[emoji])
    
    return text_string

def emojiReplace_v2(text_string):
    emoji_dict = demoji.findall(text_string)    
    for emoji in emoji_dict.keys():
        #Making the connecting token between words a normal letter 'w' because BERT's tokenizer
        #splits on special tokens like '%' and '$'
        emoji_token = 'x'.join(re.split('\W+', emoji_dict[emoji])) + ' '
        text_string = text_string.replace(emoji, emoji_token)
        
        #Controlling for multiple emojis in a row
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
