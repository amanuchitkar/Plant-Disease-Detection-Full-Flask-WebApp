import os
import logging
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import requests

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load disease information
disease_info = pd.read_csv('disease_info.csv', encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv', encoding='cp1252')

# Load model
model = tf.keras.models.load_model('trained_plant_disease_model.keras')

# Class names from validation set
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
    'Peach___healthy', 'Pepper_bell___Bacterial_spot', 'Pepper_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

def model_prediction(image_path):
    try:
        # Use the same preprocessing as test.py
        image = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 128))
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.array([input_arr])  # Convert to batch

        # print("Preprocessed Image Shape:", input_arr.shape)  # Debugging

        # Predict
        predictions = model.predict(input_arr)
        # print("Raw Predictions:", predictions)  # Debugging

        max_index = np.argmax(predictions)
        # print("Predicted Index:", max_index)  # Debugging

        return max_index  # Return class index

    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return None


app = Flask(__name__)




@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/contact')
def contact():
    return render_template('contact-us.html')

@app.route('/index')
def ai_engine_page():
    return render_template('index.html')

@app.route('/mobile-device')
def mobile_device_detected_page():
    return render_template('mobile-device.html')

WEATHER_API_KEY = "b01595d07d4e4cc2afb130127250603"
PLANT_CONDITIONS = {
    "Tomato Blight": {"temp_min": 15, "temp_max": 30, "humidity_min": 50, "humidity_max": 80},
    "Grape___Black_rot": {"temp_min": 20, "temp_max": 25, "humidity_min": 60, "humidity_max": 80},
    # Add more plant conditions...
}
@app.route('/submit', methods=['POST'])
def submit():
    if 'image' not in request.files:
        return render_template('error.html', message="No file uploaded")
    
    image = request.files['image']
    if image.filename == '':
        return render_template('error.html', message="No selected file")

    try:
        # Save uploaded file
        uploads_dir = os.path.join('static', 'uploads')
        os.makedirs(uploads_dir, exist_ok=True)
        file_path = os.path.join(uploads_dir, image.filename)
        image.save(file_path)

        # Get prediction
        pred_idx = model_prediction(file_path)
        
        if pred_idx is None:
            return render_template('error.html', message="Prediction failed")

        # Get disease name from class names instead of CSV
        disease_name = CLASS_NAMES[pred_idx]

        # Get disease information from CSV
        try:
            # Read image for display using OpenCV
            img = cv2.imread(file_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Save processed image for display
            display_path = os.path.join(uploads_dir, 'processed_' + image.filename)
            plt.imsave(display_path, img)
            
            supplement_data = supplement_info.iloc[pred_idx]
            desc = disease_info['description'][pred_idx]
            prevent = disease_info['Possible Steps'][pred_idx]
            
            return render_template('submit.html', 
                title=disease_name,  # Taken from CLASS_NAMES
                desc=desc,  # Replace with actual logic if needed
                prevent=prevent,  # Replace with actual logic if needed
                image_url=display_path,
                sname=supplement_data['supplement name'],
                simage=supplement_data['supplement image'],
                buy_link=supplement_data['buy link'],
                filename='processed_' + image.filename
            )
            
        except IndexError as e:
            logging.error(f"Index error: {str(e)}")
            return render_template('error.html', message="Data mapping error")

    except Exception as e:
        logging.error(f"General error: {str(e)}")
        return render_template('error.html', message="Processing error")

@app.route('/market', methods=['GET', 'POST'])
def market():
    return render_template('market.html', 
                           supplement_image=list(supplement_info['supplement image']),
                           supplement_name=list(supplement_info['supplement name']), 
                           disease=list(disease_info['disease_name']), 
                           buy=list(supplement_info['buy link']))


@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    return render_template('feedback.html')

DATA_SUBMISSIONS_FILE = 'user_submitted_data.csv'
@app.route('/submit_data', methods=['POST'])
def submit_data():
    try:
        # Get form data
        disease_name = request.form.get('disease_name')
        description = request.form.get('description')
        prevention = request.form.get('prevention')

        # Handle image file
        if 'image' not in request.files:
            return "No image file uploaded", 400
        
        image = request.files['image']
        if image.filename == '':
            return "No selected image file", 400

        # Save uploaded file
        uploads_dir = os.path.join('static', 'user_uploads')
        os.makedirs(uploads_dir, exist_ok=True)
        image_path = os.path.join(uploads_dir, image.filename)
        image.save(image_path)

        # Save data to CSV
        new_data = pd.DataFrame([{
            'disease_name': disease_name,
            'description': description,
            'prevention': prevention,
            'image_path': image_path
        }])

        csv_file = 'user_submitted_data.csv'
        if os.path.exists(csv_file):
            new_data.to_csv(csv_file, mode='a', header=False, index=False)
        else:
            new_data.to_csv(csv_file, mode='w', header=True, index=False)

        return "Data submitted successfully!", 200

    except Exception as e:
        logging.error(f"Data submission error: {str(e)}")
        return "Error processing submission", 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)