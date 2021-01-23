#Making necessary imports
import pandas as pd
import numpy as np
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

#Using pandas to read .csv file
df = pd.read_csv('/Users/maurycyczajka/Downloads/news.csv')
print(df)

#Getting shape of file
print('SHAPE: \n',df.shape)
print('HEAD: \n',df.head())

#Getting lables from Dataframe
labels = df.label
print('LABELS: \n',labels.head())

#Splitting dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], labels,
test_size=0.2, random_state=7)

# Initializing TfidVectorizer with stop words in English
# Setting maximum document frequency of 0.7 (if higher, frequency will be discarded)
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

#Fitting and transforming the vectorizer on train set
tfidf_train = tfidf_vectorizer.fit_transform(X_train)

#Transfroming the vectorizer on the test set.
tfidf_test = tfidf_vectorizer.transform(X_test)

#Initializing PassiveAggressiveClassifier and fitting it on tfidf_train and y_train
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)

#Predicting on test set from TfidVectorizer and calcuating accuracy with accuracy_score()
y_pred=pac.predict(tfidf_test)
score=accuracy_score(y_test, y_pred)

# Accuracy should be >92%
accuracy = round(score*100,2)
print(f'\nAccuracy: {accuracy}%')