# Automatic Detection of Hate Speech Online Using BERT Word Embeddings

## Introduction

Hate Speech is a scourge upon the internet, especially on twitter. This project aims to develop a classifier which can reliably classify hate speech tweets. This hate speech detector has been benchmarked against other systems and has achieved admirable reslts. Placing in the top 10 in the ongoing AnalyticsVidhya Twitter sentiment problem (a hackathon which has thousands of participants) achieving an F score of 0.8267. Also, placing third on classifying the HatEval dataset among 69 participant systems, with a macro F1 score of 0.547. First among those using a neural network approach.

This project is organised for convenience in folders, each with their own descriptive README file. The process of building this system can be split into 3 phases

1.  **Text_Preprocessing** - Within this folder is detailed an exploration at the various techniques used to preprocess the noisy tweet data. Demonstrated are the effects of the text pre-processing methods themselves, as well as a method for pulling down a dataframe with up-to-date information on the most popular emojis in use worldwide and a method for giving emojis representation in BERT's vocabulary by altering the BERT vocab.txt file

2.  **Further_Pre_Training** - Detailed inside is the meticulous process where millions of in-domain tweets were extracted via tweepy API, then cleaned via text pre-processing and duplicate removal. These tweets were then used to further pre-train BERT's word embeddings so they could become more adapted to the vocabulary of an informal medium like twitter, which is strewn with slang and colloqiual terminology.

3.  **Fine_Tuning** - Included is a notebook within which various hate speech datasets are inspected and critiqued. These datasets are used widely in academia, but because the focus of this project was to develop a **reliable** hate speech detector. Special care was taken for what datasets were going to use to train the model in the fine-tuning stage. Also within this directory is the final stage of the classifier, wherein transfer learning is utilised upon the BERT word embeddings and is fine-tuned the classify hate speech using a variety of text pre-processing pipelines and model architectures. All code used to build these models is in advanced Tensorflow.

The **Raw_Data** directory contains all of the csv files of the hate speech datasets included in this study for inspection and in some cases used for Fine-Tuning and testing the classifier

The **Report** directory where an interim project description and final report are enclosed. The final report is a detailed analysis and summarisation of the findings in this study

Finally the **Meeting_Minutes** directory is a summarisation of regular meetings between the project supervisor and myself. In it are suggestions from the supervisor on how to improve the classifier and what aspects of BERT I should explore and exploit in order to gain performance.

