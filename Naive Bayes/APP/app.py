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

        encoder = joblib.load(f'model/LabelEncoder.pkl')
       
        X=[]
        tfidf_vect = joblib.load(f'model/TF-IDF.pkl')
        content = ViTokenizer.tokenize(content)
        content = gensim.utils.simple_preprocess(content)
        X.append(content) 
        X_test=list_to_doc(X)
        X_test_n=tfidf_vect.transform(X_test)
        loaded_model = joblib.load(f'model/Final_Model_Text_Class.pkl')
        Y_test = loaded_model.predict(X_test_n)
        y_test = loaded_model.predict_proba(X_test_n)

        a=np.array(y_test)

    
        # Render the form again, but add in the prediction and remind user
        # of the values they input before
        return flask.render_template('main.html',
                                     result=encoder.inverse_transform(Y_test)[0],
                                     original_input={'Chính trị xã hội':round(a[0,0],6),
                                                     'Công nghệ':round(a[0,1],6),
                                                     'Khoa học':round(a[0,2],6),
                                                     'Kinh doanh':round(a[0,3],6),
                                                     'Pháp luật':round(a[0,4],6),
                                                     'Sức khỏe':round(a[0,5],6),
                                                     'Thể thao':round(a[0,6],6),
                                                     'Văn hóa':round(a[0,7],6),
                                                     'Đời sống':round(a[0,8],6)},
                                        
                                     )

if __name__ == '__main__':
    app.run(debug=True)