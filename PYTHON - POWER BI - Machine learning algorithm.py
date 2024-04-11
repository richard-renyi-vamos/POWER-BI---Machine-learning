import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv('your_dataset.csv')

# Preprocess the data (if needed)
# Example: data = preprocess_data(data)

# Split data into features (X) and target variable (y)
X = data.drop('target_column', axis=1)
y = data['target_column']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
print('Mean Squared Error:', mse)

# Export predictions to CSV
predictions_df = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})
predictions_df.to_csv('predictions.csv', index=False)
