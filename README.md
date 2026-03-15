# ChurnLite App

Streamlit demo app for the [churnlite](https://github.com/alberto1201/churnlite) library.

## How to run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## What it does

Upload a CSV file with customer data and get churn predictions instantly.
- Trains a Logistic Regression model on 80% of the data
- Predicts churn probability for each customer
- Shows ROC-AUC score and predictions table
- Download predictions as CSV
