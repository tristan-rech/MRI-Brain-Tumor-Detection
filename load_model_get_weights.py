import numpy as np
import tensorflow as tf
import os
import pickle

script_dir = os.path.dirname(__file__)
model_dir = os.path.join(script_dir, 'models')
model_path = os.path.join(model_dir, 'brain_tumor_cnn_classifier.keras')

CNN = tf.keras.models.load_model(model_path, compile=False)
CNN.compile(optimizer=tf.keras.optimizers.Adamax(learning_rate=0.001), 
            loss='categorical_crossentropy', 
            metrics=['accuracy'])

weights = CNN.get_weights()

with open('models/CNN_weights.pkl', 'wb') as f:
    pickle.dump(weights, f)

print(weights)

with open('models/CNN_weights.pkl', 'rb') as f:
    loaded_weights = pickle.load(f)

loaded_weights = loaded_weights[0]

print(loaded_weights)

# Save the model structure
model_json = CNN.to_json()
with open('models/CNN_structure.json', 'w') as json_file:
    json_file.write(model_json)