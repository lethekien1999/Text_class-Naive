import pickle
import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib

X_data = pickle.load(open(f'data/X_data.pkl', 'rb'))

y_data = pickle.load(open(f'data/y_data.pkl', 'rb'))

# X_test = pickle.load(open(f'data/X_test.pkl', 'rb'))

# y_test = pickle.load(open(f'data/y_test.pkl', 'rb'))

#biến đổi nhãn về dạng số

from sklearn import preprocessing

encoder = preprocessing.LabelEncoder()
y_data_n = encoder.fit_transform(y_data)
# y_test_n = encoder.fit_transform(y_test)

#Biến đỗi các doc về dạng if-idf

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer         
tfidf_vect = TfidfVectorizer(analyzer='word', max_features=100000,)
print('Số lượng từ điển : ',)
tfidf_vect.fit(X_data)
# joblib.dump(tfidf_vect,"TF.pkl")

X_data_tfidf =  tfidf_vect.transform(X_data)         
# X_test_tfidf =  tfidf_vect.transform(X_test)
print(len(tfidf_vect.vocabulary_))

#Train-model

from sklearn.model_selection import train_test_split
from sklearn import metrics


#Mô hình Naive-Bayes 
from sklearn.naive_bayes import MultinomialNB
model= MultinomialNB()
   
X_train, X_val, y_train, y_val = train_test_split(X_data_tfidf, y_data_n, test_size=0.3,random_state=42)        
model.fit(X_train, y_train)            
train_predictions = model.predict(X_train)
val_predictions = model.predict(X_val)
# test_predictions = model.predict(X_test_tfidf)
# joblib.dump(model,"Text_Class.pkl")
      
print('Train accuracy: ', metrics.accuracy_score(train_predictions, y_train))
print("Validation accuracy: ", metrics.accuracy_score(val_predictions, y_val))
# print("Test accuracy: ", metrics.accuracy_score(test_predictions, y_test_n))

train_predict = model.predict(X_val)

from sklearn.metrics import classification_report
print(classification_report(y_val, train_predict, target_names=encoder.classes_,digits=6))



