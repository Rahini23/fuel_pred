import numpy as np
from flask import Flask, render_template,request
import pickle
app = Flask(__name__)
model_lr = pickle.load(open('model_lr.pkl', 'rb'))
@app.route('/')
def home():
    return render_template('resultf.html')
@app.route('/predict',methods=['POST'])
def predict():
    input=request.form['drivenKM']
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model_lr.predict(final_features)
    return render_template('resultf.html', prediction_text='fuel price for kilometer driven is :{}'.format(prediction))
if __name__ == "__main__":
    app.run(debug=True)



