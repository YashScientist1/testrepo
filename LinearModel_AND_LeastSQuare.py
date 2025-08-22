# -----------------------------------------------------------
# Import required libraries
# -----------------------------------------------------------
# numpy: library for numerical computations (linear algebra, arrays, random sampling)
# matplotlib.pyplot: library for creating visual plots
# sklearn.linear_model.LinearRegression: predefined class for fitting linear regression models
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# -----------------------------------------------------------
# 1. Generate synthetic dataset
# -----------------------------------------------------------

# np.random.seed(42)
# Context: RANDOMNESS / REPRODUCIBILITY
# Sets the seed for the NumPy random number generator.
# Why: Ensures the same random dataset is generated every time you run the code,
# so results are consistent when studying.
np.random.seed(42)

# np.random.multivariate_normal(mean, cov, size)
# Context: STATISTICS / DATA GENERATION
# Generates samples from a multivariate normal (Gaussian) distribution.
# mean -> list of means for each dimension. Here [2, 2] means data is centered at point (2, 2).
# cov -> covariance matrix. [[1, 0], [0, 1]] means variance=1 for each dimension (spread of data),
#        and covariance=0 means no correlation between X1 and X2 (independent dimensions).
# size -> number of samples to draw (how many points we generate).
X_blue = np.random.multivariate_normal(mean=[2, 2], cov=[[1, 0], [0, 1]], size=100)

# np.zeros(100)
# Context: MACHINE LEARNING LABELS
# Creates an array of 100 zeros. We assign label "0" to all BLUE class points.
y_blue = np.zeros(100)

# Generate ORANGE points in the same way but centered at (4, 4).
# mean=[4, 4] shifts the cluster to a different location.
X_orange = np.random.multivariate_normal(mean=[4, 4], cov=[[1, 0], [0, 1]], size=100)

# np.ones(100)
# Context: MACHINE LEARNING LABELS
# Creates an array of 100 ones. We assign label "1" to all ORANGE class points.
y_orange = np.ones(100)

# np.vstack((A, B))
# Context: DATA PREPARATION
# Vertically stacks arrays A and B. Here it combines BLUE and ORANGE feature sets into one dataset.
X = np.vstack((X_blue, X_orange))   # Shape (200, 2): 200 samples, each with 2 features (X1, X2)

# np.concatenate((A, B))
# Context: DATA PREPARATION
# Joins arrays end-to-end. Combines BLUE and ORANGE labels into one vector of length 200.
y = np.concatenate((y_blue, y_orange))


# -----------------------------------------------------------
# 2. Fit Linear Regression model
# -----------------------------------------------------------

# LinearRegression()
# Context: MACHINE LEARNING / PREDICTION MODEL
# Creates a Linear Regression model object.
# This model learns coefficients β (weights) that best fit the data in the least squares sense.
model = LinearRegression()

# model.fit(X, y)
# Context: MACHINE LEARNING / TRAINING
# Trains the model using dataset (X = features, y = labels).
# X -> shape (200, 2): each row is a point, each column is a feature.
# y -> shape (200,): corresponding label (0 for BLUE, 1 for ORANGE).
model.fit(X, y)

# Define our own function for clarity
def predict(X):
    # model.predict(X)
    # Context: MACHINE LEARNING / INFERENCE
    # Takes input feature(s) and returns predicted output ŷ.
    # Here ŷ is a number between 0 and 1 representing how "ORANGE" the point is.
    return model.predict(X)


# -----------------------------------------------------------
# 3. Create a grid for plotting decision boundary
# -----------------------------------------------------------

# np.linspace(start, stop, num)
# Context: NUMERICAL COMPUTING
# Generates evenly spaced numbers between start and stop.
# Used to cover the entire range of X1 and X2 values in the dataset.
x_range = np.linspace(X[:,0].min()-1, X[:,0].max()+1, 200)  # 200 points across X1
y_range = np.linspace(X[:,1].min()-1, X[:,1].max()+1, 200)  # 200 points across X2

# np.meshgrid(x_range, y_range)
# Context: PLOTTING / GRID CREATION
# Takes two 1D ranges and creates two 2D grids (xx, yy).
# Each (xx[i,j], yy[i,j]) pair corresponds to a point in the 2D plane.
xx, yy = np.meshgrid(x_range, y_range)

# np.c_[A, B]
# Context: DATA RESHAPING
# Concatenates columns A and B side-by-side.
# ravel(): flattens 2D arrays into 1D.
# Together: convert xx, yy into a (N, 2) array of all grid points.
grid = np.c_[xx.ravel(), yy.ravel()]

# model.predict(grid)
# Context: MACHINE LEARNING / INFERENCE
# Predicts values for every point on the grid.
preds = model.predict(grid)

# preds.reshape(xx.shape)
# Context: ARRAY RESHAPING
# Reshapes 1D predictions back into a 2D array so we can plot contours.
zz = preds.reshape(xx.shape)


# -----------------------------------------------------------
# 4. Plot everything
# -----------------------------------------------------------

# plt.figure(figsize=(8,6))
# Context: PLOTTING
# Creates a new plot with specified width=8 and height=6 (inches).
plt.figure(figsize=(8,6))

# plt.contourf(X, Y, Z, alpha, cmap)
# Context: PLOTTING
# Creates filled contour regions based on predictions.
# X, Y -> mesh grid
# Z -> predicted labels (zz > 0.5 gives ORANGE=1 region)
# alpha=0.3: transparency
# cmap="coolwarm": colormap (blue to red)
plt.contourf(xx, yy, zz > 0.5, alpha=0.3, cmap="coolwarm")

# plt.contour(X, Y, Z, levels, colors)
# Context: PLOTTING
# Draws contour lines. Here we draw a black line where ŷ = 0.5 (decision boundary).
plt.contour(xx, yy, zz, levels=[0.5], colors="black")

# plt.scatter(x, y, c, label)
# Context: PLOTTING
# Draws scatter plot of training data.
# X_blue[:,0] -> X1 coordinate, X_blue[:,1] -> X2 coordinate, c="blue": color, label: legend text
plt.scatter(X_blue[:,0], X_blue[:,1], c="blue", label="BLUE (0)")
plt.scatter(X_orange[:,0], X_orange[:,1], c="orange", label="ORANGE (1)")

# plt.xlabel("X1"), plt.ylabel("X2"), plt.title("...")
# Context: PLOTTING
# Adds axis labels and title for better readability.
plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Linear Regression for Classification (Figure 2.1 Style)")

# plt.legend()
# Context: PLOTTING
# Displays the legend with labels (BLUE and ORANGE).
plt.legend()

# plt.show()
# Context: PLOTTING
# Displays the final plot.
plt.show()