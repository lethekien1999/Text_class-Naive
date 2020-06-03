import pickle
import pandas as pd 
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from pyvi import ViTokenizer, ViPosTagger
import gensim 
import os
import pickle  
from sklearn import preprocessing

y_data = pickle.load(open(f'data/y_data.pkl', 'rb'))
encoder = preprocessing.LabelEncoder()
y_data_n = encoder.fit_transform(y_data)


def list_to_doc(X_data):
	x =[]
	for doc in X_data:
		x.append(' '.join(doc))
	return x
# xây dựng stopwords
stop_word=[]
f = open("stopwords.txt", 'r', encoding="utf-8")
text = f.read()
for word in text.split() :
	stop_word.append(word)

X=[]
tfidf_vect = joblib.load('TF.pkl')
doc = open("XH_NLD_ (3732).txt", 'r', encoding="utf-16")
content=doc.read()
content = ViTokenizer.tokenize(content)           #tach tu 
content = gensim.utils.simple_preprocess(content) # xoa cac ki tu dac biet
sentences=[]
for word in content:
	if(word not in stop_word):
		if ("_" in word) or (word.isalpha() == True):
			sentences.append(word)

X.append(sentences) 
X_test=list_to_doc(X)

X_test_n=tfidf_vect.transform(X_test)

loaded_model = joblib.load('Text_Class.pkl')

Y_test = loaded_model.predict(X_test_n)
y_test = loaded_model.predict_proba(X_test_n)
# print(encoder.inverse_transform(Y_test))
a=np.array(y_test)
a=a*100
# df=pd.DataFrame()
# df['class']=encoder.classes_
# df['rate']=a.T
print(content)
print(X)


