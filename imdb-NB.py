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
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.metrics import accuracy_score




# clean data

def review_to_words( raw_review ):
    review_text = BeautifulSoup(raw_review).get_text()     
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    words = letters_only.lower().split()                             
    stops = set(stopwords.words("english"))                  
    meaningful_words = [w for w in words if not w in stops]   
    return( " ".join( meaningful_words )) 




# text preprocessing for training set

def text_preprocess_train(train):
    print("Text prepocessing for training set...\n")
    num_reviews = train["review"].size
    clean_train_reviews = []

    for i in range( 0, num_reviews ):
        if( (i+1)%1000 == 0 ):
            print("Review %d of %d" % ( i+1, num_reviews ))
        clean_train_reviews.append( review_to_words( train["review"][i] ) )

    # bag-of-word
    vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, max_features = 5000) 
    train_features = vectorizer.fit_transform(clean_train_reviews)

    # TF-IDF
    tfidf_transformer = TfidfTransformer()
    train_features_tfidf = tfidf_transformer.fit_transform(train_features)
    train_features_tfidf = train_features_tfidf.toarray()

    print(type(train_features_tfidf), len(train_features_tfidf))
    
    print("Bag of words...\n")
    vocab = vectorizer.get_feature_names()
    dist = np.sum(train_features_tfidf, axis=0)
    for tag, count in zip(vocab, dist):
        print(count, tag)

    return train_features_tfidf, vectorizer




# naive bayes

def NB(features, labels):
    print("****************************\n")
    print("Training naive bayes model...\n")
    clf = BernoulliNB()
#     clf = GaussianNB()
    clf.fit(features, labels)
    
    predicted =  clf.predict(features)
    print ("Accuracy on training set: {}%".format(accuracy_score(labels, predicted)*100))
    return clf




# text preprocessing on test set

def text_preprocess_test(test, vectorizer):
    print("****************************\n")
    print("Test preprocessing for test set ...\n")
    num_reviews = len(test["review"])
    clean_test_reviews = [] 
    for i in range(0,num_reviews):
        if( (i+1) % 1000 == 0 ):
            print("Review %d of %d" % (i+1, num_reviews))
        clean_review = review_to_words( test["review"][i] )
        clean_test_reviews.append(clean_review)

    test_features = vectorizer.transform(clean_test_reviews)
    test_features = test_features.toarray()
    
    tfidf_transformer = TfidfTransformer()
    test_features_tfidf = tfidf_transformer.fit_transform(test_features)
    test_features_tfidf = test_features_tfidf.toarray()
    
    print(type(test_features_tfidf), len(test_features_tfidf))
    
    return test_features_tfidf




# evluate on test set

def evluate(test_features, test_labels, myNB):
    print("****************************\n")
    print("Evaluate on test set ...\n")
    
    predicted = myNB.predict(test_features)
    
    print ("Accuracy on test set: {}%".format(accuracy_score(test_labels, predicted)*100))
    print("Classification report : \n", metrics.classification_report(test_labels, predicted))
    print("Confusion Matrix : \n", metrics.confusion_matrix(test_labels, predicted))
    
    return




# Load data and split into training set and test set
df = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
train, test = train_test_split(df, test_size=0.3, random_state=0)
test = test.reset_index()
train = train.reset_index()

# text preprocessing on training set
train_features , vectorizer = text_preprocess_train(train)
      
# model training
myNB = NB(train_features, train["sentiment"])

# evaluate on test set
test_features = text_preprocess_test(test, vectorizer)
evluate(test_features, test["sentiment"], myNB)



