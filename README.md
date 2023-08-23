# CLV

def calculate_clv(average_yearly_revenue, customer_lifespan, churn_rate):
    clv = (average_yearly_revenue * customer_lifespan) - (churn_rate * average_yearly_revenue * customer_lifespan)
    return clv

average_yearly_revenue = 1000
customer_lifespan = 5
churn_rate = 0.2  # (20% annual churn rate)

clv = calculate_clv(average_yearly_revenue, customer_lifespan, churn_rate)
print "The Customer Lifetime Value is: Rs %s" % clv



import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the data from the Excel file
df = pd.read_excel('/content/output.xlsx')

# Define the feature variables and the target variable
X = df[['Lifespan (years)', 'Average Yearly Revenue', 'Churn Rate', 'Discount Rate']]
y = df['CLV']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create a Linear Regression model
model = LinearRegression()

# Train the model using the training data
model.fit(X_train, y_train)

# Use the model to make predictions on the test data
y_pred = model.predict(X_test)

# Calculate the root mean squared error and r-squared value for the predictions on the test set
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print "RMSE: %s" % rmse
print "R2: %s" % r2

# Sample values for prediction
sample_values = [[5, 7000, 0.05, 0.07]]

# Use the model to predict the CLV for the sample values
sample_prediction = model.predict(sample_values)

print "Predicted CLV for the sample values: %s" % sample_prediction[0]
