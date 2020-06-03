import flask
import pickle
import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from pyvi import ViTokenizer, ViPosTagger
import gensim 
import os
from sklearn import preprocessing
import numpy as np

# Initialise the Flask app
app = flask.Flask(__name__, template_folder='templates')

def list_to_doc(X_data):
    x =[]
    for doc in X_data:
        x.append(' '.join(doc))
    return x

# Set up the main route
@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        # Just render the initial form, to get input
        return(flask.render_template('main.html'))
    
    if flask.request.method == 'POST':
        # Extract the input
        content = flask.request.form['text_sum']

        y_data = pickle.load(open(f'model/y_data.pkl', 'rb'))
        encoder = preprocessing.LabelEncoder()
        y_data_n = encoder.fit_transform(y_data)
       
        X=[]
        tfidf_vect = joblib.load('TF.pkl')
        content = ViTokenizer.tokenize(content)
        content = gensim.utils.simple_preprocess(content)
        X.append(content) 
        X_test=list_to_doc(X)
        X_test_n=tfidf_vect.transform(X_test)
        loaded_model = joblib.load('Text_Class.pkl')
        Y_test = loaded_model.predict(X_test_n)
        y_test = loaded_model.predict_proba(X_test_n)

        a=np.array(y_test)
        a=a*100
    
        # Render the form again, but add in the prediction and remind user
        # of the values they input before
        return flask.render_template('main.html',
                                     result=encoder.inverse_transform(Y_test)[0],
                                     original_input={'Chinh tri Xa hoi':a[0,0],
                                                     'Cong nghe':a[0,1],
                                                     'Dien anh':a[0,2],
                                                     'Giao duc':a[0,3],
                                                     'Khoa hoc':a[0,4],
                                                     'Kinh te':a[0,5],
                                                     'Phap luat':a[0,6],
                                                     'The thao':a[0,7],
                                                     }
                                     )

if __name__ == '__main__':
    app.run(debug=True)