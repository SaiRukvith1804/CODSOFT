import pandas as pd
import re
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Function to load data from a .txt file
def load_txt_data(file_path):
    try:
        # Load train data
        train_data = pd.read_csv(
            file_path,
            delimiter=':::',
            names=['Title', 'Genre', 'Description'],
            engine='python'
        )
        return train_data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# Define the file path for the .txt file
file_path = "C:\\Users\\sairu\\Downloads\\python vscodes\\CODSOFT\\Task_2_Movie_Genre_Classification\\DATASET\\train_data.txt"

# Load the data from the .txt file
train_data = load_txt_data(file_path)

# Check if data is loaded correctly
if train_data is not None:
    print("First 5 rows of the training dataset:")
    print(train_data.head())

    # Drop rows with missing plot summaries or genres in the training dataset
    train_data = train_data.dropna(subset=['Description', 'Genre'])

    # Function to Clean Text Data
    def clean_text(text):
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
        text = text.lower()  # Convert to lowercase
        return text

    # Apply the Cleaning Function to the Descriptions
    train_data['clean_description'] = train_data['Description'].apply(clean_text)

    # Feature Extraction with TF-IDF
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X = tfidf_vectorizer.fit_transform(train_data['clean_description'])
    y = train_data['Genre']

    # Split Data into Training and Testing Sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Logistic Regression Model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Make Predictions
    y_pred = model.predict(X_test)

    # Evaluate the Model
    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
    plt.title("Confusion Matrix")
    plt.show()

    # Predict on New Data
    new_plot = ["A young wizard learns about his magical powers and battles dark forces."]
    new_plot_clean = [clean_text(plot) for plot in new_plot]
    new_plot_tfidf = tfidf_vectorizer.transform(new_plot_clean)
    predicted_genre = model.predict(new_plot_tfidf)
    print("\nPredicted Genre for New Plot:", predicted_genre[0])

else:
    print("Failed to load data from .txt file.")
