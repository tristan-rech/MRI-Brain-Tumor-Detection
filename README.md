# MRI-Brain-Tumor-Detection
### Written by Tristan Pedro

## Overview
Currently, there is just a notebook, named tumor-classification-cnn.ipynb, that details the development of a deep learning model classifying brain tumors from a MRI brain image dataset. It includes steps from preprocessing and loading data, constructing and training a Convolutional Neural Network (CNN), to evaluating and visualizing the models performance. There are future plans to expand this into a website where predictions are made to user uploaded images, and the test set.

## Necessary Imports
- System Modules: os and itertools
- Image Processing: PIL.Image and cv2(OpenCV)
- Data Handling: Pandas and NumPy
- Visualization: matplotlib and seaborn
- Machine Learning: sklearn
- Deep Learning: tensorflow and keras

## Data Preprocessing
- Training & Test Data: Images and labels are loaded from [this](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) Kaggle dataset.
- Data Generators: "ImageDataGenerator" is used for preparing data for model training (augmentation).

## Model Structure
- A sequential CNN model with multiple Convolution2D and MaxPooling2D layers is constructed.
- The model's final layer has neurons equal to the number of classes, using a 'softmax' activation function for multi-class classification.
- The model is compiled with Adamax optimizer and categorical cross-entropy loss.

## Model Evalutation
- The trained model is evaluated on the training, validation, and test sets.
- Accuracy and loss for each set are displayed in a consolidated format.
- A confusion matrix and classification report provide detailed insights into the model's performance across different classes.

## Model Predictions
- The model's predictions are showcased with probabilities for a few test images.
- A group plot visually displays images from each class with the model's prediction and the probability of the predicted class.

## Saving & Loading the Model (Future Implementations)
- The model is saved in the TensorFlow/Keras format for later use.
- Instructions shown for loading the model for independent use. This step is key for showing its functionality to be implemented in the later web application.

## Key Takeaways
- The model demonstrates high accuracy in classifying brain tumors into categories: glioma, meningioma, no tumor, and pituitary.
- Visualizations and reports indicate the model's strengths and areas for potential improvement.
- The pipeline showcases the application of deep learning in medical image analysis, emphasizing the importance of accurate and detailed preprocessing, model design, and evaluation.