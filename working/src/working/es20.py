import numpy as np
import matplotlib.pyplot as plt
import h5py
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split



filename = '/home/leo/astroML_data/sample_2e7_design_precessing_higherordermodes_3detectors.h5'
f = h5py.File(filename, 'r')

print(f.keys())
print(np.len(f['det']))

X = f['mtot'][:]
labels = f['det'][:]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
# Create a logistic regression model
model = LogisticRegression(solver='lbfgs', max_iter=1000, multi_class='multinomial')
# Fit the model to the training data
model.fit(X_train.reshape(-1, 1), y_train)
# Evaluate the model on the test data
accuracy = model.score(X_test.reshape(-1, 1), y_test)
print(f"Accuracy: {accuracy:.2f}")
# Plot the decision boundary
plt.figure(figsize=(8, 6))
plt.scatter(X_train, y_train, color='blue', label='Training data', alpha=0.5)
plt.scatter(X_test, y_test, color='red', label='Test data', alpha=0.5)
x = np.linspace(X.min(), X.max(), 1000).reshape(-1, 1)
y_pred = model.predict(x)
plt.plot(x, y_pred, color='black', label='Decision boundary')       

plt.show()

