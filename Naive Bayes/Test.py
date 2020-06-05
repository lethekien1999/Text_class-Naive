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
import matplotlib.pyplot as plt

encoder=joblib.load(f'APP/model/LabelEncoder.pkl')


def list_to_doc(X_data):
	x =[]
	for doc in X_data:
		x.append(' '.join(doc))
	return x

X=[]
tfidf_vect = joblib.load(f'APP/model/TF-IDF.pkl')
path="XH_NLD_ (3732).txt"
doc = open(path, 'r')
content=doc.read()
content = ViTokenizer.tokenize(content)           #tach tu 
content = gensim.utils.simple_preprocess(content) # xoa cac ki tu dac biet

X.append(content)  

X_test=list_to_doc(X)

X_test_n=tfidf_vect.transform(X_test)

loaded_model = joblib.load(f'APP/model/Final_Model_Text_Class.pkl')

Y_test = loaded_model.predict(X_test_n)
y_test = loaded_model.predict_proba(X_test_n)
# print(encoder.inverse_transform(Y_test))
a=np.array(y_test)
a=a*100
# df=pd.DataFrame()
b=encoder.classes_
class_tl=[]
for a1 in a[0]:
	class_tl.append(a1)
# df['rate']=a.T
# plt.bar(a,b,color='blue')
# plt.show()
print(encoder.classes_)

