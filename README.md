Dog Breed Classifier

## https://dog-breed-classifier-zkxe.onrender.com
This project consists of two main components: 
a Jupyter Notebook for training and evaluating a dog breed classification model, 
and a Streamlit application for predicting dog breeds using the trained model.


Colab Notebook: Dog_breed_classifier.ipynb


Overview


The Jupyter Notebook is used for data preprocessing, model training, and evaluation. It leverages deep learning techniques to classify images of dogs into specific breeds.


Key Steps

Data Preprocessing:

Load and preprocess the dataset, which includes resizing images and normalizing pixel values.

Split the dataset into training and validation sets.


Model Training:

Define a Convolutional Neural Network (CNN) architecture using Keras.

Train the model on the training dataset while validating on the validation set.

Save the trained model as dog_breed.h5.


Evaluation:

Evaluate the model's performance using metrics such as accuracy and loss.

Visualize training history to analyze model performance over epochs.



Dataset


The dataset used for training includes images of various dog breeds.

 For demonstration purposes, the notebook focuses on three breeds: Pomeranian, Entlebucher, and Scottish Deerhound.


Streamlit App: main_app.py


Overview


The Streamlit application provides a user-friendly interface for predicting the breed of a dog from an uploaded image using the trained model.



Key Features



User Interface:


Users can upload an image of a dog in PNG format.

Prediction:


The uploaded image is processed and resized to the required input dimensions for the model.


The model predicts the breed, and the result is displayed on the interface.


How to Run



Ensure you have the necessary libraries installed: numpy, streamlit, opencv-python, and keras.



Place the trained model file (dog_breed.h5) in the same directory as main_app.py.



Run the Streamlit app using the command:



streamlit run main_app.py

Open the provided local URL in a web browser to access the app.


Classes


The model predicts among three dog breeds: Pomeranian, Entlebucher, and Scottish Deerhound.


Conclusion


This project demonstrates a complete workflow from training a deep learning model to deploying it in a web application for real-time predictions.


 The combination of Jupyter Notebook and Streamlit provides an effective way to develop and showcase machine learning applications. 


This README provides a concise explanation of the project's components and instructions on how to use them. 


Adjust the content as needed based on specific project details or additional features.
