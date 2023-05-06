import pandas as pd

# Import the necessary libraries for machine learning
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Load the employee performance dataset (assuming it is stored in a CSV file)
data = pd.read_csv(r"D:\8. Practicals\AI\Practical 6 (Expert System)\employee_performance.csv")

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.drop("performance", axis=1), data["performance"], test_size=0.2)

# Train a decision tree classifier on the training data
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Get the input values for employee performance
sales = float(input("Enter the sales value: "))
attendance = float(input("Enter the attendance value: "))
quality = float(input("Enter the quality value: "))

# Evaluate the employee performance using the machine learning model
result = clf.predict([[sales, attendance, quality]])

# Print the result
print("The employee's performance is:", result[0])
