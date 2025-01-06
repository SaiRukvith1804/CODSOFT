# Movie Genre Classification Using Machine Learning

## Overview
This project implements a **Movie Genre Classification System** using **Natural Language Processing (NLP)** and **Machine Learning** techniques. It predicts the genre of a movie based on its description using a **Logistic Regression Model**.

## Dataset
The dataset used in this project contains movie titles, genres, and descriptions stored in a `.txt` file. Each line of the file includes movie details separated by ':::' delimiters.

## Features
- **Data Loading**: Reads data from a `.txt` file and processes it into structured format.
- **Text Preprocessing**: Cleans descriptions by removing special characters and converting text to lowercase.
- **Feature Extraction**: Uses **TF-IDF Vectorization** to convert text data into numerical features.
- **Model Training**: Trains a **Logistic Regression** model to classify movie genres.
- **Evaluation Metrics**: Measures accuracy, precision, recall, F1-score, and visualizes confusion matrices.
- **Genre Prediction**: Provides predictions for new movie descriptions.

## Libraries Used
- **Pandas** - Data handling and preprocessing.
- **Scikit-learn** - Machine learning algorithms and evaluation metrics.
- **Matplotlib** - Visualization of confusion matrices.

## Workflow
1. **Data Loading**: Load data from `.txt` file.
2. **Data Cleaning**: Preprocess text to remove unwanted characters.
3. **Text Vectorization**: Convert text descriptions into numerical features using **TF-IDF**.
4. **Model Training**: Train a **Logistic Regression** classifier.
5. **Model Evaluation**: Assess performance using classification reports and confusion matrices.
6. **Prediction**: Predict genres for new movie descriptions.

## Results
- **Logistic Regression Model**: Achieved high accuracy and reliable performance for genre classification.

## Requirements
- Python 3.x
- Required Libraries: pandas, scikit-learn, matplotlib

## Example Input and Output
```
Predicted Genre for New Plot: Fantasy
```

