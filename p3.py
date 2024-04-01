import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
# Generate some random data for demonstration
np.random.seed(0)
X = 2 * np.random.rand(100, 1)  # Generate 100 random values between 0 and 2
y = 4 + 3 * X + np.random.randn(100, 1)  # Linear relationship with some noise

# Plot the data
plt.scatter(X, y, color='blue')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Scatter Plot of Data')
plt.show()

# Fit the linear regression model
model = LinearRegression()
model.fit(X, y)

# Get the coefficients
intercept = model.intercept_[0]
slope = model.coef_[0][0]
print('Intercept:', intercept)
print('Slope:', slope)

# Plot the regression line
plt.scatter(X, y, color='blue')
plt.plot(X, model.predict(X), color='red')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression')
plt.show()
