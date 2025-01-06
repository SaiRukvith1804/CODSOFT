# Spam SMS Detection Using Machine Learning

## Overview
This project implements a **Spam SMS Detection System** using Machine Learning techniques. It classifies SMS messages as either 'Spam' or 'Ham' (Not Spam) based on their content. The system uses **Natural Language Processing (NLP)** for text preprocessing and **Machine Learning models** for classification.

## Dataset
The project utilizes a dataset containing labeled SMS messages (Spam or Ham). The dataset is loaded from an Excel file (`spam.xlsx`) and preprocessed for training and testing.

## Features
- **Text Preprocessing**: Cleans and normalizes SMS messages by removing punctuation, numbers, and extra spaces.
- **Feature Extraction**: Converts text data into numerical form using **TF-IDF Vectorization**.
- **Model Training**: Trains two models—
  1. **Naive Bayes** (MultinomialNB)
  2. **Support Vector Machine** (SVM) with linear kernel.
- **Evaluation Metrics**: Evaluates models using accuracy, classification reports, and confusion matrices.
- **Real-Time Predictions**: Accepts new SMS messages from the user and predicts whether they are spam or ham.

## Libraries Used
- **Pandas** - Data handling and preprocessing.
- **NumPy** - Numerical computations.
- **Matplotlib** & **Seaborn** - Visualization of results.
- **Scikit-learn** - Machine Learning algorithms and evaluation metrics.

## Workflow
1. **Data Loading**: Import dataset from Excel.
2. **Data Cleaning**: Handle missing values and format text.
3. **Text Processing**: Clean text and vectorize using TF-IDF.
4. **Model Training**: Train Naive Bayes and SVM models.
5. **Model Evaluation**: Assess accuracy, precision, recall, and F1-score.
6. **Visualization**: Generate confusion matrices for performance insights.
7. **User Input Prediction**: Take SMS input at runtime for classification.

## Results
- **Naive Bayes Model**: Quick and effective for spam detection.
- **SVM Model**: Provides higher accuracy with better performance on text classification.

## Requirements
- Python 3.x
- Required Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn

## Example Input and Output
```
Enter an SMS message (or type 'exit' to stop): Congratulations! You have won a lottery.
Naive Bayes Prediction: Spam
SVM Prediction: Spam
```


