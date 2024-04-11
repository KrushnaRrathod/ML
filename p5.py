import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

# Specify the file path where the CSV file is located
csv_file_path = r'C:\Users\rathod\Downloads\customer_churn.csv'

# Load the dataset from the CSV file
df = pd.read_csv(csv_file_path)

# Perform one-hot encoding for categorical variables
df_encoded = pd.get_dummies(df, columns=['gender', 'region', 'phone_service', 'internet_service', 'contract', 'payment_method'])

# Split the dataset into features (X) and target variable (y)
X = df_encoded.drop('churn', axis=1)
y = df_encoded['churn']

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Naïve Bayesian classifier
clf = GaussianNB()

# Train the classifier
clf.fit(X_train, y_train)

# Predict the target variable for the test data
y_pred = clf.predict(X_test)

# Compute the accuracy of the classifier
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
