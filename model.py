import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np

# Load your dataset
file_path = "hearts.csv"  # Replace with the path to your dataset
data = pd.read_csv(file_path)

# Define features and label
features = ['Age', 'Sex', 'ChestPainType', 'FastingBS', 'RestingBP', 'Cholesterol', 
            'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']
label_column = 'HeartDisease'

X = data[features]
y = data[label_column]

# Categorical and numerical feature separation
categorical_features = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
numerical_features = ['Age', 'FastingBS', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']

# Preprocess categorical features using OneHotEncoder
column_transformer = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_features)  # Apply OneHotEncoder to categorical features
    ],
    remainder='passthrough'  # Keep numerical features as-is
)

# Ensure that the transformed data remains a DataFrame
X_encoded = column_transformer.fit_transform(X)

# Extract transformed column names
categorical_feature_names = column_transformer.named_transformers_['cat'].get_feature_names_out(categorical_features)
all_feature_names = list(categorical_feature_names) + numerical_features

# Convert the transformed data back to a DataFrame
X_encoded = pd.DataFrame(X_encoded, columns=all_feature_names)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42)

# Initialize the Naive Bayes classifier
nb_classifier = GaussianNB()

# Train the classifier
nb_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = nb_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Print the accuracy
print(f"Accuracy of the Naive Bayes Classifier: {accuracy * 100:.2f}%\n\n\n")

# Function to predict heart disease based on user input
def predict_heart_disease(*features):
    sample = pd.DataFrame([features], columns=X.columns)  # Create a DataFrame for the input
    sample_encoded = column_transformer.transform(sample)  # Apply the same transformation as training
    prediction = nb_classifier.predict(sample_encoded)[0]
    prediction_prob = nb_classifier.predict_proba(sample_encoded)[0]
    
    prob_no = prediction_prob[0] * 100
    prob_yes = prediction_prob[1] * 100
    
    if prob_yes < 30:
        risk_level = "Low"
    elif prob_yes < 70:
        risk_level = "Moderate"
    else:
        risk_level = "High"
        
    return {
        "prediction": int(prediction),
        "probability_yes": round(prob_yes, 2),
        "probability_no": round(prob_no, 2),
        "risk_level": risk_level
    }
