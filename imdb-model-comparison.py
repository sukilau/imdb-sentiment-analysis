import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from bs4 import BeautifulSoup  
import re
import nltk
from nltk.corpus import stopwords 
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import roc_auc_score



# text preprocessing
def review_to_words( raw_review ):
    review_text = BeautifulSoup(raw_review).get_text()     
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    words = letters_only.lower().split()                             
    stops = set(stopwords.words("english"))                  
    meaningful_words = [w for w in words if not w in stops]   
    return( " ".join( meaningful_words )) 

def text_preprocess(data):
    num_reviews = len(data["review"])
    clean_data = [] 
    for i in range(0,num_reviews):
        clean_review = review_to_words( data["review"][i] )
        clean_data.append(clean_review)
    return clean_data



# Naive Bayes
def NB(train_features, train_labels, test_features, test_labels):

    # train model using pipeline (tokenizer => tf-idf transformer => linear SVM classifier)
    text_clf = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', BernoulliNB())])
    _ = text_clf.fit(train_features, train_labels)
    
    # evaluate on test set
    predicted = text_clf.predict(test_features)
    print("********** Naive Bayes Model **********")
    print ("Accuracy on test set: {}%".format(accuracy_score(test_labels, predicted)*100))
    print("Classification report : \n", metrics.classification_report(test_labels, predicted))
    print("Confusion Matrix : \n", metrics.confusion_matrix(test_labels, predicted))
    AUC = roc_auc_score(test_labels, predicted)
    print("Area Under Curve : \n", AUC)
    print("Prediction: ", text_clf.predict(["bad movie stupid story","very nice movie wonderful character", "slow pace at the beginning but a suprising ending, quite a nice movie"]))
    
    return AUC



# linear SVM
def SVM(train_features, train_labels, test_features, test_labels):

    # train model using pipeline (tokenizer => tf-idf transformer => linear SVM classifier)
    text_clf = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=5, random_state=42))])
    _ = text_clf.fit(train_features, train_labels)

    # evaluate on test set
    predicted = text_clf.predict(test_features)
    print("********** SVM Model **********")
    print ("Accuracy on test set: {}%".format(accuracy_score(test_labels, predicted)*100))
    print("Classification report : \n", metrics.classification_report(test_labels, predicted))
    print("Confusion Matrix : \n", metrics.confusion_matrix(test_labels, predicted))
    AUC = roc_auc_score(test_labels, predicted)
    print("Area Under Curve : \n", AUC)
    print("Prediction: ", text_clf.predict(["bad movie stupid story","very nice movie wonderful character", "slow pace at the beginning but a suprising ending, quite a nice movie"]))
    
    return AUC



# random forest
def RF(train_features, train_labels, test_features, test_labels):

    # train model using pipeline (tokenizer => tf-idf transformer => linear SVM classifier)
    text_clf = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', RandomForestClassifier(n_estimators = 100, verbose=1))])
    _ = text_clf.fit(train_features, train_labels)

    # evaluate on test set
    predicted = text_clf.predict(test_features)
    print("********** Random Forest Model **********")
    print ("Accuracy on test set: {}%".format(accuracy_score(test_labels, predicted)*100))
    print("Classification report : \n", metrics.classification_report(test_labels, predicted))
    print("Confusion Matrix : \n", metrics.confusion_matrix(test_labels, predicted))
    AUC = roc_auc_score(test_labels, predicted)
    print("Area Under Curve : \n", AUC)
    print("Prediction: ", text_clf.predict(["bad movie stupid story","very nice movie wonderful character", "slow pace at the beginning but a suprising ending, quite a nice movie"]))
    
    return AUC



# load data and split into training set and test set
df = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
train, test = train_test_split(df, test_size=0.3, random_state=0)
train = train.reset_index()
test = test.reset_index()

# text preprocessing
train_features = text_preprocess(train)
train_labels = train["sentiment"]
test_features = text_preprocess(test)
test_labels = test["sentiment"]

# NB
AUC_NB = NB(train_features, train_labels, test_features, test_labels)

# linear SVM
AUC_SVM = SVM(train_features, train_labels, test_features, test_labels)

# Random Forest
AUC_RF = RF(train_features, train_labels, test_features, test_labels)

print("Naive Bayes AUC : ", AUC_NB)
print("Linear SVM AUC : ", AUC_SVM)
print("Random Forest AUC : ", AUC_RF)






