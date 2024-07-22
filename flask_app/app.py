import os
import numpy as np
import cv2
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras import layers, models
import tensorflow as tf

app = Flask(__name__)



@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    #Create folder to save image uploaded
    uploads_dir = 'uploads'
    if not os.path.exists(uploads_dir):
        os.makedirs(uploads_dir)
    
    if file:
        filename = os.path.join('uploads', file.filename)
        file.save(filename)
        return jsonify({'success': 'File uploaded successfully'}), 200


@app.route('/predict_cnn/<file_name>', methods=['POST','GET'])
def predict_with_cnn(file_name): 
    print(file_name)

    # Load the trained model
    model = models.load_model('models/VGG_freeze.h5')


    class_names = ["angry","happy","ahegao","sad","neutral","surprise"]       
    
    # Process the image and make predictions
    image_size = (64,64)
    img_data = cv2.imread(os.path.join('uploads', file_name))
    img_data = cv2.resize(img_data.copy(), image_size,interpolation=cv2.INTER_AREA)
    img_data = img_data/255

    img_data = np.array([img_data])
    print(img_data.shape)

    prediction = model.predict(np.array([img_data]))
    predicted_class = class_names[np.argmax(prediction)]
    return jsonify({'predicted_class': predicted_class}), 200

if __name__ == '__main__':
    app.run(debug=True)
