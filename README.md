Initially, this project's aim is to detect the presence of hate speech with binary classification. Then furthermore, the program must describe further if the target harassed is an individual or a group and whether the message is aggressive or not.

The code is a bit disorganised at the moment, but it will be cleaned up in time. The project is organised as such:


1.  **CSC3002-Hate_Speech_Detection - Assembling and Cleaning the Fine-Tuning Data.ipynb** - I assemble what I believe to be datasets containing hate speech and inspect their content to determine if they can be reliably used to train my model. One can also view the PDF version in gitlab however it will not be as up-to-date as the notebook file present

2.  **RetrievingPretrainingData.ipynb** - I assemble pretraining data from many sources. Unlike the data I use for fine-tuning, this data is unsupervised and is not often represented in text form to begin with. So I resort to using the tweepy package to retrieve the tweets via their tweet ID. Like the fine-tuning data, I analyse and explain the data sources. As has been recommended in papers I've read, I target data which is domain related so I use tweets likely to contain abuse towards different minorities and also tweet datasets which are likely to primarily user-generated, as hate speech online is mostly user-generated

3.  **FurtherPretraining.ipynb** - I then feed the fine-tuning data and the pretraining data into a script provided by the original BERT repo to further pretrain their BERT model from it's final checkpoint using the data I assembled in my pre-training notebook as well as the data I will use to fine-tune the model. This further pre-training is exactly alike to how the original model was trained; by performing masked language modelling and next sentence prediction on each sequence.

4. **HatEval_Tensorflow_TPU.ipynb** - Finally I load the pretrained model - whether it is the original BERT model or my further pretrained model, and fine tune it to detect hate speech. Currently I have not yet developed my further pretrained model successfully, but I expcet to do so very soon. For the moment my model only is being used on the HatEval dataset as I believe it's the most reliable, however I may change this in the future.

Also present is the notebook file Unsupervised_Data_Augmentation which performs backtranslation on the sequences it is fed, however this has not proven to improve performance. I'm keeping it in the repo so I can use it, prove is doesnt improve predictive power and analyse why.

This project is still in progress, more functionality will be added very soon.