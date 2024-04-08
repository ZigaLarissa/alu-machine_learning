Sure, here's a detailed README.md file for your project:

# TV Sales Prediction API

This project aims to predict TV sales based on TV marketing expenses using a linear regression model. The model is trained on the `tvmarketing.csv` dataset and then hosted on a FastAPI server with a ngrok tunnel for public access.

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Model Training](#model-training)
   - [Linear Regression with NumPy](#linear-regression-with-numpy)
   - [Linear Regression with Scikit-Learn](#linear-regression-with-scikit-learn)
   - [Linear Regression using Gradient Descent](#linear-regression-using-gradient-descent)
4. [API Deployment](#api-deployment)
   - [FastAPI Server](#fastapi-server)
   - [ngrok Tunnel](#ngrok-tunnel)
5. [Usage](#usage)
   - [Accessing the API](#accessing-the-api)
   - [Testing the API](#testing-the-api)

## Introduction

The goal of this project is to create a predictive model that can estimate TV sales based on TV marketing expenses. The model is trained using three different approaches: NumPy, Scikit-Learn, and Gradient Descent. The best performing model is then deployed as a FastAPI server with a public URL provided by ngrok.

## Dataset

The dataset used in this project is `tvmarketing.csv`, which contains two columns: `TV` (TV marketing expenses) and `Sales` (sales amount). This dataset is located in the `data` directory of the repository.

## Model Training

The notebook file in the repository contains the code for training the linear regression model using the three different approaches mentioned above.

### Linear Regression with NumPy

The `np.polyfit()` function is used to fit a linear regression model using NumPy. The slope and intercept of the model are then used to make predictions.

### Linear Regression with Scikit-Learn

The Scikit-Learn `LinearRegression` class is used to fit a linear regression model. The model is trained on 80% of the data and tested on the remaining 20%. The Root Mean Squared Error (RMSE) is calculated for the model.

### Linear Regression using Gradient Descent

A custom implementation of gradient descent is used to train the linear regression model. The cost function and its partial derivatives are defined, and the gradient descent algorithm is implemented to find the optimal slope and intercept.

## API Deployment

### FastAPI Server

The trained linear regression model is deployed as a FastAPI server. The server has a single endpoint `/predict` that accepts a JSON request with a `tv` field containing the TV marketing expenses, and returns the predicted sales amount.

### ngrok Tunnel

The FastAPI server is exposed to the public using an ngrok tunnel. The public URL provided by ngrok can be used to access the API.

## Usage

### Accessing the API

To access the TV sales prediction API, you can use the public URL provided by ngrok, plus an endpoint of `/predict`. The URL will be in the format `https://<random-string>.ngrok.io/predict`.

### Testing the API

You can use any platform of your choice, such as Postman, to test the API. Send a POST request to the `/predict` endpoint with a JSON payload containing the TV marketing expenses, and the API will return the predicted sales amount.

Example JSON payload:

```json
{
  "tv": 27.1
}
```

Example response:

```json
134.04
```

This means that the model predicts a sales amount of 27.1 for a TV marketing expense of 134.04.


### How to run the program:
T o run the program on your laptop, 1. Clone the repository
```
clone 
```
2. Open the notebook, choose a kettle to run with, then click on Run All.
3. Remember to interupt the second last code cell, in order to run the last cell that will give you the Public Url to test with.

### Contributor:
- Larissa Bizimungu: l.bizimungu@alustudent.com
