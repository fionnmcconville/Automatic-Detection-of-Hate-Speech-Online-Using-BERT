# Further Pre-Training Stage

Within this directory we query millions tweets from various tweet ID datasets in RetrievingPretrainingData.ipynb for further pre-training of the BERT embeddings and run further pre-training in the `FurtherPretraining.ipynb` notebook using the custom BERT repositiory within this project. More details below...

## `RetrievingPretrainingData.ipynb`

The Tweepy library is utilised along with a twitter developer API to source these tweets which come from hand-picked tweet ID datasets that are as within-domain to the subject of hate speech as possible. The content of these datasets are elaborated upon in the notebook. 

The tweet ID datasets are stored in a virtual GCS bucket and the resulting text of the datasets is stored in the GCS bucket as well, albeit in a separate directory

The datasets processed for collecting tweet IDs range from between ~20,000 tweet IDs to ~16,000,000 (although only 3,000,000 of these tweets were attempted to be retrieved because of tie constraints and because I didn't want the content of that dataset to dominate the others too much as I wanted the furth-pretraining stage to be as exposed to as wide a variety of subjects as possible)

The magnitude of some datasets meant that the function for querying the tweet ID dataset had to be run all day, uniterrupted in a separate session for maximum return. Checkpoints every 500,000 tweets were implemented to ensure that tweet text already collected was stored in the GCS bucket when the session did eventually time out mid-function (this occured not due to errors in the function but due to the computer science lab not having PCs that ran all day)

## `FurtherPretrainingData.ipynb`
**Must be opened in Google Colab (a link to this notebook in Google Colab is at the top of the notebook if ne chooses to open the notebook in another environment)**

The tweet text obtained from the previous notebook is then used for further pre-training the BERT embeddings in this notebook. This is so the embeddings become more accustomed to the parlance of users online and can observe unusual text such as pre-processed emojis (whether that is in words or tokens) and segmneted hashtags - the way the text of tweets are structured after undergoing this sort of pre-processing is very different to how it would be under normal circumstances (such as wikipedia, which is the original corpus that BERT was pre-trained upon)

Crucially, one can choose to pre-process the tweet text however they like, so as to match the pre-processing utilised later on donwstream fine-tuning and so maximise performance. The preprocessing methods are fetched by cloning this repo into the virtual colab space and navigating to the Text_Preprocessing directory wherein the `loadData()` function containing all of the methods for pre-processing is located. Choosing a preprocessing pipeline is very easy for the user.

After preprocessing, many tweets are removed from the dataset. This is because text pre-processing crucially removes the RT handle along with converting mentions to common tags. After this, the textual content of each retweeted tweet is identical and so accordingly removed.

The next two scripts of `create_pretraining_data.py` and `run_pretraining` are scripts that were sourced from the original BERT repo. However they have been forked to this project and modified to not include the next sentence prediction task, the learning from which is irrelevant to text classification tasks.

`create_pretraining_data.py` converts each tweet into feature formats that BERT can understand. Tokenizing the text and mapping each wordpiece token to an ID representation from the BERT vocab file. The difference between this script and the original script is that all segment IDs are initialised to the same value of 0, so there is no next sentence prediction attemmpted in the `run_pretraining` stage.

`run_pretraining.py` executes further pretraining on the original BERT embeddings using the tweet data converted to features in the previous `create_pretraining_data` script. Crucially next sentence prediction loss is removed so this task has no impact on learning, even if it were to run in the background unknowingly.

 The only task that has any importance or effect in this stage is the masked language modelling task, which is beneficial to natural langugage understanding and can vastly improve performance upon downstream tasks, fine-tuning upon data that is within-domain to the data used in further pre-training.
