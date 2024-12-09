import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the dataset
df = pd.read_csv('Toddler Autism dataset July 2018.csv')  # Replace with the actual path

# Preprocess categorical columns
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
df['Ethnicity'] = le.fit_transform(df['Ethnicity'])
df['Jaundice'] = le.fit_transform(df['Jaundice'])

# Encode the target variable
df['Class/ASD Traits '] = le.fit_transform(df['Class/ASD Traits '])

# Define features and target
X = df[['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10']]  # Updated to include 10 features
y = df['Class/ASD Traits ']  # Target column

# Scale the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save the trained model to a .pkl file
joblib.dump(model, 'autism_detector_model.pkl')
print("Model saved as autism_detector_model.pkl")
