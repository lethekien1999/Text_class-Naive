import pickle
import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

X_data = pickle.load(open(f'data/X_data.pkl', 'rb'))

y_data = pickle.load(open(f'data/y_data.pkl', 'rb'))


X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.1,random_state=42) 

#biến đổi nhãn về dạng số

encoder = preprocessing.LabelEncoder()
encoder.fit(y_train)
# joblib.dump(encoder,"LabelEncoder.pkl")
y_train_n = encoder.transform(y_train)
y_val_n= encoder.transform(y_val)

report = open("Thamso.txt",'w')

for n in range(1,3,1):
	print("running on min_df = "+str(n))
	for k in range(4,6,1):
		h = 0.05*k
		print("running on max_df = "+str(h))
		tfidf_vect=TfidfVectorizer(analyzer='word', min_df=n, max_df=h, sublinear_tf=True )
		tfidf_vect.fit(X_train)
		X_train_tfidf= tfidf_vect.transform(X_train)
		X_val_tfidf= tfidf_vect.transform(X_val)

		for i in range(1,10,1):

			j=0.1*i
			print("running on alpha = "+str(j))
			# Mô hình Naive Bayes
			model=MultinomialNB(alpha=j)
			# Train
			model.fit(X_train_tfidf,y_train_n)

			train_pre= model.predict(X_train_tfidf)
			val_pre=model.predict(X_val_tfidf)


			score_train = metrics.accuracy_score(train_pre,y_train_n)
			score_val= metrics.accuracy_score(val_pre, y_val_n)

			report.write(f"* min_df= {n} , max_df= {h}, alpha= {j}, accuracy_train={score_train}, accuracy_val={score_val} \n")
report.close()


