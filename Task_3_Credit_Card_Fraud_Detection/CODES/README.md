# Credit Card Fraud Detection Using Machine Learning

## Overview
This project implements a **Credit Card Fraud Detection System** using **Machine Learning** techniques. It detects fraudulent transactions based on user behavior and transaction details using a **Random Forest Classifier**.

## Dataset
The dataset contains details about credit card transactions, including attributes such as transaction amount, location, time, and user demographics. The dataset is split into training and testing sets.

## Features
- **Data Preprocessing**: Combines datasets, removes unnecessary features, and handles categorical data through one-hot encoding.
- **Feature Scaling**: Standardizes data for optimal model performance.
- **Machine Learning Models**: Implements **Random Forest Classifier** for prediction.
- **Evaluation Metrics**: Analyzes performance using accuracy, classification reports, confusion matrices, and ROC curves.
- **Fraud Prediction**: Accepts real-time user inputs to predict if a transaction is fraudulent.

## Libraries Used
- **Pandas** - Data manipulation and preprocessing.
- **NumPy** - Numerical computation.
- **Scikit-learn** - Machine learning algorithms and evaluation metrics.
- **Matplotlib & Seaborn** - Visualization of results and performance metrics.

## Workflow
1. **Data Loading**: Combine training and testing datasets.
2. **Data Cleaning**: Drop irrelevant columns and handle categorical variables.
3. **Feature Scaling**: Normalize numeric features using **StandardScaler**.
4. **Model Training**: Train a **Random Forest Classifier**.
5. **Evaluation**: Evaluate performance using confusion matrices, ROC curves, and accuracy scores.
6. **Prediction**: Predict fraudulent transactions based on user input.

## Results
- **Random Forest Model**: Achieved high accuracy and effectively identified fraudulent transactions.
- **ROC Curve Analysis**: Demonstrated excellent performance in distinguishing between legitimate and fraudulent transactions.

## Requirements
- Python 3.x
- Required Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn

## Example Input and Output
```
Enter Age: 30
Enter Income: 50000
Enter Transaction Amount: 1200
The transaction is: Fraudulent
```

