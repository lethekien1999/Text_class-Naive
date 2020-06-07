import pickle
import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from time import time

X_data = pickle.load(open(f'data/X_data.pkl', 'rb'))

y_data = pickle.load(open(f'data/y_data.pkl', 'rb'))

X_test = pickle.load(open(f'data/X_test.pkl', 'rb'))

y_test = pickle.load(open(f'data/y_test.pkl', 'rb'))



X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.1, random_state=42) 
#biến đổi nhãn về dạng số

encoder = preprocessing.LabelEncoder()
encoder.fit(y_train)
# joblib.dump(encoder,"LabelEncoder.pkl")
y_train_n = encoder.transform(y_train)
y_test_n = encoder.transform(y_test)
y_val_n = encoder.transform(y_val)
#Biến đỗi các doc về dạng if-idf
         
tfidf_vect = TfidfVectorizer(analyzer='word', min_df=1, max_df=0.15, sublinear_tf= True)
tfidf_vect.fit(X_train)
# joblib.dump(tfidf_vect,"TF-IDF.pkl")

X_train_tfidf =  tfidf_vect.transform(X_train)
X_val_tfidf =tfidf_vect.transform(X_val)
X_test_tfidf = tfidf_vect.transform(X_test)         


from sklearn import metrics

#Mô hình Naive-Bayes 
from sklearn.naive_bayes import MultinomialNB
model= MultinomialNB(alpha=0.02)
   
#train model       
time1 = time()  
model.fit(X_train_tfidf, y_train_n)    
time2 = time()        
train_predictions = model.predict(X_train_tfidf)
time3=time()
test_predictions = model.predict(X_test_tfidf)
time4=time()
val_predictions = model.predict(X_val_tfidf)

# joblib.dump(model,"Final_Model_Text_Class.pkl")
      
print('Train accuracy: ', metrics.accuracy_score(train_predictions, y_train_n))
print('Test accuracy: ', metrics.accuracy_score(test_predictions, y_test_n))
print('Val accuracy :', metrics.accuracy_score(val_predictions,y_val_n))


from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_train_n, train_predictions, target_names=encoder.classes_,digits=4))
print(classification_report(y_test_n, test_predictions,target_names=encoder.classes_,digits=4))
print(classification_report(y_val_n, val_predictions,target_names=encoder.classes_,digits=4))


print(confusion_matrix(y_test_n,test_predictions))
print("thời gian train",time2-time1)
print(" thời gian đưa ra đánh giá" ,time4-time3)