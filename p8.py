from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score, KFold, ShuffleSplit, RepeatedKFold, LeaveOneOut
from sklearn.svm import SVC

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Initialize the SVM classifier
svm = SVC()

# K-Fold Cross Validation
print("K-Fold Cross Validation:")
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
kfold_scores = cross_val_score(svm, X, y, cv=kfold)
print("Scores:", kfold_scores)
print("Mean Accuracy:", kfold_scores.mean())

# Shuffled K-Fold Cross Validation
print("\nShuffled K-Fold Cross Validation:")
shuffle_split = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
shuffle_scores = cross_val_score(svm, X, y, cv=shuffle_split)
print("Scores:", shuffle_scores)
print("Mean Accuracy:", shuffle_scores.mean())

# Repeated K-Fold Cross Validation
print("\nRepeated K-Fold Cross Validation:")
repeated_kfold = RepeatedKFold(n_splits=5, n_repeats=2, random_state=42)
repeated_kfold_scores = cross_val_score(svm, X, y, cv=repeated_kfold)
print("Scores:", repeated_kfold_scores)
print("Mean Accuracy:", repeated_kfold_scores.mean())

# Leave-One-Out Cross Validation
print("\nLeave-One-Out Cross Validation:")
leave_one_out = LeaveOneOut()
leave_one_out_scores = cross_val_score(svm, X, y, cv=leave_one_out)
print("Scores:", leave_one_out_scores)
print("Mean Accuracy:", leave_one_out_scores.mean())
