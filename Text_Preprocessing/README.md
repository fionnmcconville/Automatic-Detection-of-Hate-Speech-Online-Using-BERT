# Text Pre-processing Stage

## `Text_Preprocesssing.ipynb`

This notebook demonstrates each text-preprocessing technique that was experimented with throughout this project. Tweets from the HatEval dataset are used as examples. Within the notebook, the methods that are experimented with are:

* **Hashtag Segmentation** - The wordSegment library is used to segment words which are combined together via n gram language based algorithms , i.e #BuildThatWall -> Build That Wall. Demonstrateably improves performance on classification

* **Emoji Translation** - This function is implemented via the emoji library on pypi. The emojis can either be replaced by words or via a unique token. This approach is elaborated upon in the report and in the notebook itself. Both functions improve performance on text classification.

* **Lemmatize Text** - Much like stemming but more complex. Unlike stemming which heuristically chops off words without taking into account the conext with which it is used, lemmatizing returns words that actually exists in the dictionary. This pre-processing however was not beneficail to performance however, this could be because BERTwas not pre-trained upon lemmatized text and already has adequate enough representation for words that end in -ed or -ing because of it's wordpiece tokenization.

* **Removing Stopwords** - Stopwords are words that are deemed to not give much info, semantically neutral words such as 'the' and 'with' are removed. Removing stopwords was quite deliterious to model performance, possibly because BERT was pre-trained on text that did not have stopwords removed.

* **Removing Punctuation** - Slight benefit to performance. Removes punctuation and special characters as advertised - must be implemented *after* hashtag segmentation.

Also included is the baseline `preprocess()` function. This method implements the traditional techniques associated with pre-processing noisy tweet text such as as replacing mentions with common tags ,  removing URLs, punctuation removal, removing retweet handles and converting to text to lowercase.



## `New_Vocab_File_For_Emojis`

Within this notebook, the top 500 emojis are parsed from the html of http://www.emojistats.org/ into a csv file (the HTML of this site is located within this directory but can be downloaded again for a more up-to-date listing of the most common emojis). These emojis are then converted into unique tokens using `emojiReplace_v2()` and replace the unused tokens in a copy of BERT's vocab file. Individual as well as plural representations of the top 500 emojis are stored within the vocab file.

Putting these tokens in the vocab file - (which is then uploaded to the GCS bucket where the BERT model is stored) allows BERT to have tokenized embeddings for each of these emojis. These embedding weights can either become updated during fine-tuning only or they can be updated in further pre-training and fine-tuning.

After extensive experimentation though, it has been discovered that converting emojis to words is a better option.


## `preprocessing.py`

All of the preprocessing functions are located within this python file. Within the file is the function loadData() which is imported into the notebooks where text pre-processing is required (before running further pre-training and before fine-tuning) and the user can choose at will which pre-processing pipeline they wish at ease within those notebooks.
