# Customer Churn Prediction Project

This project implements an end-to-end customer churn prediction workflow for the IBM Telco Customer Churn dataset.

## Objective

Predict whether a customer will churn (Yes / No) using customer attributes such as tenure, contract type, payment method, internet service, and charges.

## What is included

- Data understanding and preparation
- Data cleaning and preprocessing
- Exploratory data analysis with visualizations
- Feature engineering using meaningful churn indicators
- Decision tree model development and evaluation
- Model interpretation and feature importance review
- Saved model pipeline for future predictions
- Flask API endpoint for churn prediction

## Project structure

- data/ - copied assignment dataset
- notebook/ - Jupyter exploration and model notebook
- model/ - saved trained model pipeline
- app.py - Flask API for inference
- requirements.txt - Python dependencies
- sample_request.json - example request payload
- train_churn_model.py - training and model serialization script


## Setup

```bash
pip install -r requirements.txt
```

## Train the model

```bash
python train_churn_model.py
```

## Run the API

```bash
python app.py
```

Use the endpoint:

```text
POST http://localhost:5000/predict
```

Example request body is available in sample_request.json.

## Example response

```json
{
  "prediction": "Yes",
  "churn_probability": 0.82
}
```

## Notes

The solution uses a preprocessing pipeline with one-hot encoding for categorical values and a decision-tree-based model trained with a fixed random seed for reproducibility. The final model is saved as a reusable pickle pipeline in the model folder.
