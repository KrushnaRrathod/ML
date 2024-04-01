from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics

# Load the Iris dataset
iris = load_iris()

# Split the dataset into features (X) and target variable (y)
X = iris.data
y = iris.target

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the SVM classifier
svm = SVC()

# Train the classifier
svm.fit(X_train, y_train)

# Predict the target variable for the test data
y_pred = svm.predict(X_test)

# Compute the accuracy of the classifier
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
