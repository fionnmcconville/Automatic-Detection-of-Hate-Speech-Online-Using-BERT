# Fine-Tuning Stage

Within this directory, several well known hate speech datasets are inspected for content. There is a strict definition of hate speech that guides this project and it must be ensured that these datasets are reliable. This extensive investigation takes place in `Assembling_and_Cleaning_the_Fine_Tuning_Data`.

After this the model is compiled in `Running_BERT_Tensorflow_TPU.ipynb`. The user can choose at will what methods and techniques they'd like o employ when building the model

## `Assembling_and_Cleaning_the_Fine_Tuning_Data.ipynb`

A brief Description of all datasets investigated in this notebook is below:

**HatEval 2019** – From SemEval-2019 shared task 5. The targets of hate speech in this dataset are either women or immigrants.  The dataset was annotated from non-expert annotators from the crowdsourcing platform Figure Eight (F8), then the tweets were further reviewed by two more expert annotators. According to F8 there an intercoder-agreement score of 83% in the hate speech labelling of the dataset. 13K tweets overall (10K training, 3K testing), 4210 labelled HS

**OffensEval 2019** - From SemEval-2019 shared task 6. This task was for categorizing offensive language on twitter rather than hate speech. There were three columns to describe the data: if the data was offensive, if it was a targeted insult/threat or not and if the target was a group, individual or other. Annotated by experienced annotators from the crowdsourcing platform Figure 8 (F8), the hypothesis was that if an entry were offensive, targeted and directed towards an individual/group then perhaps it could qualify as hate speech. 13.2K tweets overall, 3481 (potential) labelled HS

**ICVSM 2017** - Hate speech dataset collected through filtering tweets that contained terms in the HateBase – a crowdsourced hate speech lexicon. To avoid false positives that occurred in prior work  which considered all uses of particular terms as hate speech, annotators were instructed not to make their decisions based upon any words or phrases in particular, no matter how offensive, but on the overall tweet and the inferred context. The intercoder-agreement score provided by CF was 92%. 25K tweets overall tweets, 1430 labelled HS

**ICVSM 2018** - Tweets in this set were labelled one of four categories: normal, abusive, hateful  and spam. Annotated by users on Crowdflower, it is the largest by far of all datasets considered for the fine-tuning stage, it has 100K tweets, 4965 labelled HS.

**Waseem and Hovy 2016** - Tweet ID datasets . Datasets consisted of tweets labelled as either sexist, racist, both or neither. The tweets were collected by filtering twitter API to attain tweets that had one of 17 terms. The tweets were reviewed by the authors for HS, then further reviewed by experts . Most of the racism tweets are targeting Muslim people and most of those that are considered sexist are criticising contestants on an Australian cooking show. [41]

**AnalyticsVidhya.com Practice Problem** – Very little information on the methodology followed for collection or annotation of tweet data.  According to the website “For the sake of simplicity, we say a tweet contains hate speech if it has a racist or sexist sentiment associated with it. So, the task is to classify racist or sexist tweets from other tweets.” The dataset is part of an ongoing competition so results can be benchmarked against other systems, also test data labels are unavailable prior to evaluation, so an honest evaluation of model performance can be realized.

## `Running_BERT_Tensorflow_TPU.ipynb`
**Must be opened in Google Colab (a link to this notebook in Google Colab is at the top of the notebook if one opens the notebook in another environment)**

The user can train and predict on the Hateval set, the AnalyticsVidhya.com set or a combined set that was created in `Assembling_and_Cleaning_the_Fine_Tuning_Data.ipynb` for offensive or hate speech data. They may also choose if they elect to use further pre-trained embeddings. (Although saying this the BERT model as well as the further pre-trainied embeddings are in GCS storage, which is not publicly available).

The user can choose within this notebook what text preprocessing pipeline they'd like to undergo. Each text preprocessing method is demonstrated in detail in the `Text_Preprocessing.ipynb` notebook in the text preprocessing directory

Furthermore, the user can choose what fine tuning architecture they'd like to appened to the end attention layers in BERT. In particular they can elect to use a stacked biLTSM model or a multi-layer perceptron, instead of the default implementation on BERT's fine-tuning stage which is originally a feed-forward layer with a softmax operation. 

One can also modify the fine tuning hyperparameters at ease, the dropout probability can be increased, training batch size and initial learning rate can be altered. Also, the amount of layers or hidden neurons can be changed if it's applicable. Also the pooled output of BERT's CLS token, which is used for classification, can be normalised. This is to solve the exploding gradient problem, which BERT models suffer from on occasion.

You can use a nested cross validation loop for evaluation. This was necessary so that reliable results could be ensured as sometimes, due to Tensorflow's stochastic nature, the variance of metrics run-to-run could exceed the performance gin in metrics that were garnered from improvements to the model.

An early stopping estimator function for TPUs is also implemented. The standard train_and_evaluate function, as of writing, does not have early stopping support for distributed training with TPUs. Therefore a custom function had to be improvised which does this. The training and evaluation does not happen concurrently in memory however. Each time training resumes, the large BERT model must be loaded back into memory which can take a while.

Also a brief error report is included, one can inspect false positives, false negatives and observe which words seem to correlate with each set.