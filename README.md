# Restaurant-Rating-Prediction


This project is aimed at predicting the ratings of restaurants based on various factors using machine learning techniques. It utilizes a dataset containing information about restaurants, such as location, cuisine type, average cost, and more, to train a model that can accurately predict the ratings of new restaurants.

Table of Contents
Introduction
Dataset
Dependencies
Installation
Usage
Model Training
Evaluation
Contributing
License
Introduction
The Restaurant Rating Prediction project aims to provide a solution for estimating the ratings of restaurants based on their features. By leveraging machine learning algorithms and a comprehensive dataset, the project allows users to input restaurant information and receive a predicted rating as an output.

This repository contains the necessary code and resources to train the prediction model and evaluate its performance. It also provides an interface for users to interact with the trained model and generate predictions for new restaurant data.

Dataset
The project utilizes a dataset of restaurant information, including various features such as:

Restaurant ID
Restaurant Name
Location
Cuisine Type
Average Cost
Online Delivery
Table Booking
Rating
The dataset is stored in CSV format and is located in the data directory. It is used for both training the model and evaluating its performance.

Dependencies
The following dependencies are required to run the project:

Python 3.6+
Pandas
NumPy
Scikit-learn
Matplotlib
Flask (for the web interface)
Please make sure to have these dependencies installed before running the code.



Usage
To use the project, follow these steps:

Ensure that the dataset file (data/restaurant_data.csv) is available.
Train the prediction model by running the train_model.py script.
Once the model is trained, you can use the web interface to generate predictions by running the app.py script.
Access the web interface by opening your browser and visiting http://localhost:5000.
Enter the required information about the restaurant in the provided form and submit it.
The predicted rating will be displayed on the web page.
Model Training
To train the prediction model, run the train_model.py script. This script uses the dataset (data/restaurant_data.csv) to train a machine learning model that can predict restaurant ratings.

During training, the script performs the following steps:

Loads the dataset.
Preprocesses the data, including handling missing values, encoding categorical variables, and splitting into training and testing sets.
Trains a machine learning model using the training data.
Evaluates the trained model's performance using the testing data.
Saves the trained model as a file (model.pkl) for future use.
Evaluation
The evaluation of the model's performance is done during the training process in the train_model.py script. The script calculates various evaluation metrics, such as mean absolute error (MAE), mean squared error (MSE), and R-squared, to assess the accuracy of the model's predictions.

These evaluation metrics provide insights into how well the model performs in estimating the restaurant ratings. The results are displayed in the console after training.


