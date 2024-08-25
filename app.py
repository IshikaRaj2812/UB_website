from flask import Flask, render_template, request, jsonify, url_for
from flask_cors import CORS
import numpy as np
import joblib
import logging
import math as Math

app = Flask(__name__, static_url_path='/static', static_folder='static')
CORS(app)

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Load the model and scalar
try:
    model = joblib.load('gpr_model.pkl')
    scalar = joblib.load('scalar.pkl')
    
    model_v = joblib.load('gpr_model_crack.pkl')
    scalar_v = joblib.load('scalar_crack.pkl')
    logging.info("Model and scalar loaded successfully")
except Exception as e:
    logging.error(f"Error loading model or scalar: {e}")
    raise

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        logging.debug(f"Received data: {data}")

        # Extract features
        f1 = 0 if data['dp'] == 0 or data['bw']==0 else (data['Aps'] * data['fse'] )/(data['bw'] *data['dp'])
        f2 = 0 if data['dp'] == 0 or data['bw']==0 else (data['Asl'] * data['fyl']) / (data['bw'] * data['dp'])
        f3 = data['bfb']
        f4 = data['dp']
        f5 = 0 if data['s'] == 0 or data['bw']==0 else (data['Asv'] * data['fyv']) / (data['s']*data['bw'])
        f6 = data['bw']
        f7 = np.sqrt(data['f_c'])
        f8 = 0 if data['dp'] == 0 else data['a'] / data['dp']
        f9 = data['Aps'] * data['fD']* Math.sin(data['α'])
        features = np.array([[f1, f2, f3, f4, f5, f6,f7,f8,f9]])
        logging.debug(f"Calculated features: {features}")

        
        # Extract features for V
        f1_v = data['Aps'] * data['fse']
        f2_v = data['Asl'] * data['fyl']
        f3_v = data['bfb']
        f4_v = data['dp']
        f5_v = 0 if data['s']==0 else (data['Asv'] * data['fyv'])/data['s']
        f6_v = f6
        f7_v = f7
        f8_v = f8
        f9_v = f9
        f10_v = data['Wcr']
        features_v = np.array([[f1_v, f2_v, f3_v, f4_v, f5_v, f6_v, f7_v, f8_v, f9_v, f10_v]])
        # Scale features
        X_scaled = scalar.transform(features)
        X_scaled_v = scalar_v.transform(features_v)
        logging.debug(f"Scaled features: {X_scaled}")

        # Make prediction
        prediction, variance = model.predict(X_scaled, return_std=True)
        prediction_v, variance_v = model_v.predict(X_scaled_v, return_std=True)
        logging.debug(f"Raw prediction: {prediction}, Raw variance: {variance}")

        # Transform prediction and variance
        final_prediction = float(np.exp(prediction[0]))
        final_variance = float(np.exp(variance[0]))
        final_prediction_v = float(np.exp(prediction_v[0]))
        final_variance_v = float(np.exp(variance_v[0]))
        V_Vn=final_prediction_v/final_prediction
        logging.info(f"Final prediction: {final_prediction}, Final variance: {final_variance}")

        return jsonify({
            'prediction': final_prediction,
            'variance': final_variance,
            'prediction_v': final_prediction_v,
            'variance_v': final_variance_v,
            'V_Vn': V_Vn
        })

    except Exception as e:
        logging.error(f"Error in prediction: {e}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) 