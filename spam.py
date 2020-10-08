# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 02:18:49 2020

@author: Asus
"""
import re
import pandas as pd
import nltk

msg = pd.read_csv(r'E:\S-4\CSOC\spamClassifier\SMSSpamCollection', sep='\t', names=["label", "message"])
msg.head()
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle
ps=PorterStemmer()
corpus = []

for i in range(0, len(msg)):
    review = re.sub('[^a-zA-Z]', ' ', msg['message'][i])
    review = review.lower() 
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000)
X =cv.fit_transform(corpus).toarray()
pickle.dump(cv, open('cv-transform.pkl','wb'))
'''

from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer(max_features=5000)
X =cv.fit_transform(corpus).toarray()
'''

y=pd.get_dummies(msg['label'])
y=y.iloc[:,1];

from sklearn.model_selection import train_test_split
X_train, X_test, y_train,y_test= train_test_split(X,y,test_size=0.20, random_state=0)

from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(X_train,y_train)

y_pred=spam_detect_model.predict(X_test)


from sklearn.metrics import confusion_matrix
confusion_m=confusion_matrix(y_test,y_pred)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)



pickle.dump(spam_detect_model, open('modelPickle.pkl','wb'))