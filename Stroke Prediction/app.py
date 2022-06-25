from flask import Flask ,render_template ,request
import joblib
import pickle
import numpy as np
import os

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('Stroke Prediction.html')

@app.route("/result",methods = ['POST','GET'])
def result():
    gender=int(request.form['gender'])
    age=int(request.form['age'])
    hypertension=int(request.form['hypertension'])
    heart_disease = int(request.form['heart_disease'])
    ever_married = int(request.form['ever_married'])
    work_type = int(request.form['work_type'])
    Residence_type = int(request.form['Residence_type'])
    avg_glucose_level = float(request.form['avg_glucose_level'])
    bmi = float(request.form['bmi'])
    smoking_status = int(request.form['smoking_status'])

    x=np.array([gender,age,hypertension,heart_disease,ever_married,work_type,Residence_type,
                avg_glucose_level,bmi,smoking_status]).reshape(1,-1)

    scaler_path=os.path.join('C:/Users/adity/Machine Learning Projects/Stroke Prediction/Models','scaler.pkl')
    scaler=None
    with open(scaler_path,'rb') as scaler_file:
        scaler=pickle.load(scaler_file)

    x=scaler.transform(x)

    model_path=os.path.join('C:/Users/adity/Machine Learning Projects/Stroke Prediction/Models','model.sav')
    dt=joblib.load(model_path)

    Y_pred=dt.predict(x)

    # for No Stroke Risk
    if Y_pred==0:
        return render_template('Stroke Prediction.html', prediction_text=' You are safe ' , )
    else:
        return render_template('Stroke Prediction.html', prediction_text=' Stroke Risk : Please consult a doctor  ')

if __name__=="__main__":
    app.run(debug=True,port=7384)