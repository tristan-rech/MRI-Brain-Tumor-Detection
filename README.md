# MRI-Brain-Tumor-Detection
### Tristan Pedro

## Overview of Project
This project begins with a Jupyter Notebook (`tumor-classification-cnn.ipynb`) where I preprocess an MRI brain image dataset and dive into why deep learning, especially CNNs, works well for this kind of problem. The notebook walks through building and tuning a CNN model, showing how it's great for image classification, especially with medical images like MRIs.

The choice of deep learning was driven by its proven capability in handling complex image recognition tasks, and CNNs were selected due to their effectiveness in image classification problems, particularly in identifying patterns and features in medical imaging.

Following the development and fine-tuning of the CNN model in the notebook, this project extends to the realm of practical application through a web interface. Built with Flask, the web application leverages the trained CNN model to provide real-time predictions on pre-loaded MRI images (subset of the test set). This integration not only showcases the model's capabilities in a user-friendly format but also demonstrates the potential of deep learning models in real-world applications.

## Features Created
- **Jupyter Notebook**: Comprehensive analysis and model development with data preprocessing, model training, and evaluation.
- **Web Application**: A Flask-based web application where users can view predictions on a subset of the test set of MRI images. The application uses the Convolutional Neural Network (CNN) trained in the Jupyter Notebook for tumor classification.

## Technologies Used
- **Backend**: Flask (Python web framework)
- **Machine Learning**: TensorFlow, Keras
- **Image Processing**: PIL (Python Imaging Library)
- **Data Handling and Visualization**: Pandas, NumPy, Matplotlib, Seaborn
- **Others**: Logging, Exception Handling

## Data Preprocessing
- **Data Acquisition**: The MRI images and their labels were obtained from [this Kaggle dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset). It provides a diverse set of brain images, crucial for training a robust model.
  
- **Data Augmentation**: To enhance the model's ability to generalize and to mitigate overfitting, I used TensorFlow's `ImageDataGenerator`. This tool allowed for augmenting the images in various ways (like rotation, zoom, flip) to artificially expand the dataset.

- **Normalization and Resizing**: Each image was resized to a standard dimension and normalized to ensure uniformity in the input data, which is important for effective training of the CNN.

- **Train-Test Split**: The dataset was split into training and test sets. The training set is used to train the model, while the test set helps in evaluating its performance on unseen data.


## Model Structure
The CNN model used in this project is a sequential model composed of several Convolution2D and MaxPooling2D layers, carefully structured for effective image classification:

- **Convolution Layers**: The model includes multiple `Conv2D` layers with 128 and 256 filters. These layers are responsible for extracting features from the images. The first two convolution layers have 128 filters each, followed by another set of two with the same number. The next four convolution layers have 256 filters each.
  
- **Pooling Layers**: `MaxPooling2D` layers are used after certain convolution layers to reduce the spatial dimensions (width and height) of the output volume, helping to reduce the number of parameters and computation in the network.

- **Output Layer**: The final layer of the model is a Dense layer with neurons equal to the number of classes, using a 'softmax' activation function for multi-class classification. This allows the model to output a probability distribution over the classes.

- **Optimization and Loss Function**: The model is compiled with the Adamax optimizer and categorical cross-entropy loss function. This combination is chosen for effective learning and generalization in multi-class classification tasks.

- **Parameters and Size**: The total number of parameters in the model is 3,763,940 (14.36 MB). All these parameters are trainable, ensuring that the model can learn complex patterns in the data.

This structure is designed to effectively capture the intricate patterns in MRI brain images, leading to accurate classification of brain tumors.


## Model Evalutation
- The trained model is evaluated on the training, validation, and test sets.
- Accuracy and loss for each set are displayed in a consolidated format.
- A confusion matrix and classification report provide detailed insights into the model's performance across different classes.

## Model Predictions
- The model's predictions are showcased with probabilities for a few test images.
- A group plot visually displays images from each class with the model's prediction and the probability of the predicted class.

## Saving & Loading the Model 
- The model is saved in the TensorFlow/Keras format for later use.

## Key Takeaways for Model Training
- **High Accuracy**: The model demonstrates high accuracy in classifying brain tumors into categories such as glioma, meningioma, no tumor, and pituitary. This is indicative of the model's ability to effectively learn from the training data.
- **Model Architecture's Impact**: The specific architecture of the CNN, with its multiple convolution and pooling layers, played a crucial role in this high accuracy. The depth and complexity of the model allowed it to capture intricate features in the MRI images, which is essential for accurate classification in medical imaging.
- **Importance of Preprocessing and Design**: The project underscored the importance of thorough preprocessing and careful model design. Proper image augmentation, normalization, and resizing were key in preparing the data for successful model training.
- **Insights from Visualizations**: Visualizations and performance reports were instrumental in identifying the model's strengths and areas for improvement. They provided a clear understanding of how well the model was performing and where it could be refined.
- **Real-World Application and Challenges**: The pipeline showcases the practical application of deep learning in medical image analysis, highlighting both the potential and the challenges in this field, such as dealing with complex data and ensuring the model's robustness and reliability.


## Key Takeaways for Web Application Development
### Overcoming Deployment Challenges
- **Deployment on Render.com**: Successfully deploying the Flask application on render.com required navigating through various challenges. This experience provided valuable insights into the nuances of deploying web applications on cloud platforms.
- **Resource Limitations**: The constraints of the free version of render.com highlighted the need for efficient resource management, particularly for memory-intensive operations like loading a deep learning model.

### Planned Features
- **User Upload Functionality**: One of the main features under consideration is allowing users to upload their own MRI images for analysis. However, this feature may necessitate upgrading from the free tier of the web service to handle additional resource requirements.

### Learning and Debugging
- **Debugging with Logging**: Implementing and refining logging mechanisms was crucial for identifying and troubleshooting issues during development and after deployment.
- **Performance Optimization**: Balancing the application's performance with resource limitations was a key aspect of the development process, especially for real-time model predictions.

### Future Improvements
- **Enhanced User Experience**: Plans to further refine the user interface for better interactivity and user engagement.
- **Scalability Considerations**: Exploring scalability options to ensure that the application can handle increased loads, especially if the user upload feature is implemented.

### Insights into Web Development with Flask
- **Flask's Flexibility and Capability**: Working with Flask showcased its versatility as a web framework, suitable for both small-scale projects and more complex web applications.
- **Integration with Machine Learning Models**: Integrating TensorFlow/Keras models into a Flask application demonstrated the practicality of using Python for end-to-end development, from model training to deployment.

## What I Learned
The development of this project has been both challenging and rewarding, offering me valuable insights in various areas:

- **End-to-End Development**: Gained hands-on experience in managing the complete lifecycle of a project. This involved everything from the initial idea to the final implementation, covering machine learning model creation and web application development.

- **Problem-Solving and Debugging**: Enhanced my problem-solving abilities, particularly in debugging complex systems. A significant part of this learning involved understanding and implementing logging to effectively track and resolve issues.

- **Cloud Deployment**: Learned the intricacies of cloud deployment, especially while working with the free version of Render.com's web service. This involved navigating resource limitations and finding ways to optimize performance under constraints.

- **UI/UX Design**: Improved my skills in user interface and experience design. The constraints imposed by the free version of the hosting service made it crucial to implement a user interface that could effectively visualize loading states, significantly aiding user experience.

- **Adaptability and Learning**: The project underscored the importance of adaptability and continuous learning in the fast-paced and ever-changing field of technology. It reinforced the idea that facing and overcoming new challenges is a constant in this field.

## Running Notebook & Web Application Locally

This section outlines the steps to run the Flask application locally and how to work with the Jupyter Notebook.

### Prerequisites
- Python 3.11
- pip (Python package manager)
- Jupyter Notebook (for running the `tumor-classification-cnn.ipynb` file)
- Access to the internet for downloading the dataset (if wanting to run the Jupyter Notebook)

### Running the Jupyter Notebook
1. **Download the Dataset**:
   - The dataset for the Jupyter Notebook is not included in the GitHub repository. You need to download it from [Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset).
2. **Place the Dataset**:
   - After downloading, extract the dataset and place it in the designated folder within the project directory.
3. **Open the Notebook**:
   - Navigate to the notebook directory and launch Jupyter Notebook:
     ```
     jupyter notebook
     ```
   - Open the `tumor-classification-cnn.ipynb` file in the Jupyter interface.

### Setting Up the Flask Application
1. **Clone the Repository**:
    ```
    git clone https://github.com/tripedro/MRI-Brain-Tumor-Detection.git
    ```

2. **Navigate to the Project Directory**:
    ```
    cd MRI-Brain-Tumor-Detection
    ```

3. **Install Dependencies**:
- Ensure all required Python packages are installed:
  ```
  pip install -r requirements.txt
  ```

### Running the Flask Application
1. **Start the Flask Server**:
    ```
    python app.py
    ```
2. **Access the Application**:
- Open a web browser and go to `http://127.0.0.1:5000` (or what it tells you to go to) to view the application.

Follow these steps to set up and run the MRI-Brain-Tumor-Detection project on your local machine for development and testing purposes.
