from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
import pandas as pd
import logging

app = Flask(__name__)


model = joblib.load('pipeline_xgb_model.pkl')
label_encoder_cut = joblib.load('label_encoder_cut.pkl')
label_encoder_color = joblib.load('label_encoder_color.pkl')
label_encoder_clarity = joblib.load('label_encoder_clarity.pkl')


logging.basicConfig(level=logging.DEBUG)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
   
        carat = float(request.form['carat'])
        cut = request.form['cut']
        color = request.form['color']
        clarity = request.form['clarity']
        depth = float(request.form['depth'])
        table = float(request.form['table'])
        x = float(request.form['x'])
        y = float(request.form['y'])
        z = float(request.form['z'])

        
        cut_encoded = label_encoder_cut.transform([cut])[0]
        color_encoded = label_encoder_color.transform([color])[0]
        clarity_encoded = label_encoder_clarity.transform([clarity])[0]

       
        feature_names = ['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'x', 'y', 'z']
        input_data = pd.DataFrame([[carat, cut_encoded, color_encoded, clarity_encoded, depth, table, x, y, z]], 
                                  columns=feature_names)


        predicted_price = model.predict(input_data)

        return render_template('index.html', prediction_text=f"The predicted price for the diamond is: ${predicted_price[0]:.2f}")

    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")  # Log the error for debugging
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
