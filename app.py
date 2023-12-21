from flask import Flask, request, jsonify, render_template, url_for
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import os
import random

# intialize app
app = Flask(__name__, static_folder='mri-images')

# directory where this script is located
script_dir = os.path.dirname(__file__)
# path to the model directory
model_dir = os.path.join(script_dir, 'models')
# path to the model file
model_path = os.path.join(model_dir, 'brain_tumor_cnn_classifier.keras')
print(model_path)
# load model
CNN = tf.keras.models.load_model(model_path, compile=False)
CNN.compile(optimizer=tf.keras.optimizers.Adamax(learning_rate=0.001), 
            loss='categorical_crossentropy', 
            metrics=['accuracy'])

# function for retrieving prediction from model given an image path
def get_model_prediction(image_path):
    # load and preprocess the image
    img = Image.open(image_path).resize((224, 224))

    # convert grayscale images to RGB
    if img.mode != 'RGB':
        img = img.convert('RGB')

    img_array = np.expand_dims(np.array(img), axis=0) # add batch dimension

    # predict using the CNN model
    prediction = CNN.predict(img_array)

    # interpret the prediction
    predicted_index = np.argmax(prediction[0])
    class_labels = ['glioma', 'meningioma', 'no tumor', 'pituitary']
    predicted_class = class_labels[predicted_index]

    return predicted_class

# load html template
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get-random-image', methods=['GET'])
def get_random_image():
    # select a random directory and then a random image within the image directory
    class_dirs = ['glioma', 'meningioma', 'notumor', 'pituitary']
    selected_class = random.choice(class_dirs)
    image_dir = os.path.join('mri-images', selected_class)
    image_name = random.choice(os.listdir(image_dir))
    image_path = os.path.join(image_dir, image_name)

    # call prediction function to recieve predicted label
    predicted_label = get_model_prediction(image_path) 

    web_accessible_image_path = url_for('static', filename=f'{selected_class}/{image_name}')

    return jsonify({
        'image_path': web_accessible_image_path,
        'actual_label': selected_class,
        'predicted_label': predicted_label
    })

if __name__ == '__main__':
    app.run(debug=True)
