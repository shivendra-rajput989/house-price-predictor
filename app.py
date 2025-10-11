from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('house_price_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        area = float(request.form['area'])
        bedrooms = int(request.form['bedrooms'])
        age = int(request.form['age'])

        prediction = model.predict(np.array([[area, bedrooms, age]]))[0]
        return render_template('index.html', result=f'Predicted Price: ₹{prediction:.2f} Lakhs')

    except Exception as e:
        return render_template('index.html', result=f'Error: {e}')

if __name__ == '__main__':
    app.run(debug=True)
