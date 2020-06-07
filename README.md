Phân loại văn bản sử dụng Naive Bayes

Dữ liệu tham khảo từ : https://github.com/duyvuleo/VNTC/tree/master/Data 
Bộ dữ liệu gồm 9 thể loại :
Chính trị xã hội, Công nghệ, đời sống, khoa học, kinh doanh, pháp luật, sức khỏe, văn hóa, thể thao
tập Train :26261 văn bản
tập test : 18000 văn bản

-----------------------------------------------------

Run file TienXuLy.py --> tập dữ liệu đã được xử lý ( tách từ,xóa ký tự đặc biệt) lưu dưới dạng file pkl trong thư mục /data

--------------------------------------------------------
Sau khi đã có dữ liệu file pkl trong data 
 Run NaiveBayes.py ---> 3 mô hình được lưu dưới dạng pkl bao gồm:
LabelEncoder.pkl : lưu mô hình chuyển nhãn về dạng số
TF-IDF.pkl : lưu mô hình chuyển văn bản về dạng TF-IDF
Final_Model_Text_Class.pkl : mô hình dự đoán Naive Bayes đã được train

-----------------------------------------------------------
Coppy 3 mô hình trên vào APP/model/
Run file app.py ta sẽ được:
Serving Flask app "app" (lazy loading)
 * Environment: production
   WARNING: This is a development server. Do not use it in a production deployment.
   Use a production WSGI server instead.
 * Debug mode: on
 * Restarting with stat
 * Debugger is active!
 * Debugger PIN: 114-835-955
 * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)


Vào địa chỉ http://127.0.0.1:5000/ trên trình duyệt web và sử dụng chương trình
