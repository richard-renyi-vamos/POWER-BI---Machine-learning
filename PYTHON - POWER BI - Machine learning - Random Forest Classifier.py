# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load data into Power BI
dataset = pd.DataFrame(data)

# Split data into features and target variable
X = dataset.drop('target_column', axis=1)
y = dataset['target_column']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Random Forest classifier
clf = RandomForestClassifier()

# Train the model
clf.fit(X_train, y_train)

# Make predictions
predictions = clf.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, predictions)

# Output the accuracy score
print("Accuracy:", accuracy)
