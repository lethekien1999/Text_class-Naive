from pyvi import ViTokenizer, ViPosTagger
import gensim 
import os
import pickle

def getData(folder_path):
    X = []
    y = []


    # Tên Thư mục là nhãn 
    categories = os.listdir(folder_path)  #danh sach ten thu muc
    for category in categories:
        cate_path = os.path.join(folder_path,category) 
        documents = os.listdir(cate_path) #danh sach ten van ban 
        for document in documents:
            doc_path = os.path.join(cate_path,document) 
            document = open(doc_path,'r', encoding="utf-16")
            contentDoc = document.read()
            contentDoc = ViTokenizer.tokenize(contentDoc)           #tach tu 
            contentDoc = gensim.utils.simple_preprocess(contentDoc) # xoa cac ki tu dac biet

            X.append(contentDoc)
            y.append(category)
    return X,y

def list_to_doc(X_data):
  x =[]
  for n in X_data:
    x.append(' '.join(n))
  return x
  
# Xử lý tập Train
train_path ='Data Train'
X_data, y_data = getData(train_path)
X_data = list_to_doc(X_data)
pickle.dump(X_data, open('data/X_data.pkl', 'wb'))
pickle.dump(y_data, open('data/y_data.pkl', 'wb'))

# Xử lý tập Test
test_path ='Data Test'
X_test, y_test = getData(test_path)
X_test = list_to_doc(X_test)
pickle.dump(X_test, open('data/X_test.pkl', 'wb'))
pickle.dump(y_test, open('data/y_test.pkl', 'wb'))


