import pandas as pd
import numpy as np

from bs4 import BeautifulSoup  
import re

import nltk
from nltk.corpus import stopwords 
import nltk.data

import logging
from gensim.models import word2vec
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import roc_auc_score



# load data

df = pd.read_csv( "labeledTrainData.tsv", header=0, delimiter="\t", quoting=3 )
train, validation = train_test_split(df, test_size=0.3, random_state=42)

test = pd.read_csv( "testData.tsv", header=0, delimiter="\t", quoting=3 )
unlabeled_train = pd.read_csv( "unlabeledTrainData.tsv", header=0, delimiter="\t", quoting=3 )

print("Read %d labeled training set, %d labeled validation set, %d labeled test set, and %d unlabeled training set\n" % (train["review"].size, validation["review"].size, test["review"].size, unlabeled_train["review"].size ))




# text preprocessing 

# note: Word2Vec works better with stopwords and numbers
def review_to_wordlist(review, remove_stopwords=False):
    review_text = BeautifulSoup(review, 'lxml').get_text()
    review_text = re.sub("[^a-zA-Z0-9]"," ", review_text)
    words = review_text.lower().split()
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    return(words)




# NLTK's punkt tokenizer (to split a review into parsed sentences)

# note: input of Word2Vec takes a list of sentences
# nltk.download()
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
def review_to_sentences(review, tokenizer, remove_stopwords=False):
    raw_sentences = tokenizer.tokenize(review.strip())
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append( review_to_wordlist( raw_sentence,remove_stopwords ))
    return sentences


sentences = []
print("Parsing sentences from training set")
for review in train["review"]:
    sentences += review_to_sentences(review, tokenizer)

print("Parsing sentences from unlabeled set")
for review in unlabeled_train["review"]:
    sentences += review_to_sentences(review, tokenizer)
    
print(type(sentences), len(sentences), sentences[0])




# Word2Vec

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)

num_features = 300                       
min_word_count = 40                         
num_workers = 4       
context = 10                                                                                          
downsampling = 1e-3 

print("Training Word2Vec model on labeled and unlabeled training set ...")
model = Word2Vec(sentences, workers=num_workers,size=num_features, min_count = min_word_count,                 window = context, sample = downsampling)

model.init_sims(replace=True)
model.save("300features_40minwords_10context")
# model = Word2Vec.load("300features_40minwords_10context")

print(type(model.wv.syn0), len(model.wv.syn0), model.wv.syn0[0])
print(type(model.wv.index2word), len(model.wv.index2word),model.wv.index2word[0:10])




# averaging feature vector

def makeFeatureVec(review, model, num_features):
    featureVec = np.zeros((num_features,),dtype="float32")
    nwords = 0.
    index2word_set = set(model.wv.index2word) #index2word is the volcabulary list of the Word2Vec model
    for word in review:
        if word in index2word_set: 
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])
    featureVec = np.divide(featureVec,nwords)
    return featureVec

def getAvgFeatureVecs(reviews, model, num_features):
    counter = 0.
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    for review in reviews:
       if counter%1000. == 0.:
           print("Review %d of %d" % (counter, len(reviews)))
       reviewFeatureVecs[counter] = makeFeatureVec(review, model,num_features)
       counter = counter + 1.
    return reviewFeatureVecs


print("Creating average feature vectors for labeled training set")
clean_train_reviews = []
for review in train["review"]:
    clean_train_reviews.append(review_to_wordlist(review, remove_stopwords=True))
trainVector = getAvgFeatureVecs(clean_train_reviews, model, num_features)

print("Creating average feature vectors for validation set")
clean_validation_reviews = []
for review in validation["review"]:
    clean_validation_reviews.append( review_to_wordlist(review, remove_stopwords=True))
validationVector = getAvgFeatureVecs(clean_validation_reviews, model, num_features )
    
print("Creating average feature vectors for test set")
clean_test_reviews = []
for review in test["review"]:
    clean_test_reviews.append( review_to_wordlist(review, remove_stopwords=True))
testVector = getAvgFeatureVecs(clean_test_reviews, model, num_features )




# random forest

forest = RandomForestClassifier( n_estimators = 100 )

print("Training random forrest on labeled training set...")
forest = forest.fit(trainVector, train["sentiment"])




# evaluate on validation set

predicted = forest.predict( validationVector )

validationLabels = validation["sentiment"]

print("Evaluate on validation set ...\n")
print ("Accuracy on validation set: {}%".format(accuracy_score(validationLabels, predicted)*100))
print("Classification report : \n", metrics.classification_report(validationLabels, predicted))
print("Confusion Matrix : \n", metrics.confusion_matrix(validationLabels, predicted))
AUC = roc_auc_score(validationLabels, predicted)
print("Area Under Curve : \n", AUC)




# make prediction on test set
predicted = forest.predict( testVector )
output = pd.DataFrame( data={"id":test["id"], "sentiment":predicted} )
output.to_csv( "word2vec_avgvector_RF.csv", index=False, quoting=3 )


