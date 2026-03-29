# Info ---------------------------------------------------------------------------------------------
# Tutorial to implement gradient descent for linear regression in Python

# Setup --------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.datasets import fetch_california_housing

# Set print options for better readability
np.set_printoptions(suppress=True, precision=6)
pd.set_option("display.float_format", "{:.3f}".format)
pd.set_option("display.width", None)
pd.set_option("display.max_columns", None)

# Data ---------------------------------------------------------------------------------------------
# Load data
data = fetch_california_housing()
X_df = pd.DataFrame(data.data, columns=data.feature_names)
y_s = pd.Series(data.target, name=data.target_names[0])

df = pd.concat([X_df, y_s], axis=1)

# Inspect data
print(df.head())
print(df.describe().T)

# Standardize predictors with pandas
X_df = (X_df - X_df.mean()) / X_df.std(ddof=0)

# Convert to NumPy for numerical work
X = X_df.to_numpy(dtype=float)
y = y_s.to_numpy(dtype=float)

# Add intercept column
X = np.column_stack((np.ones(X.shape[0]), X))

print("\nShapes:")
print(f"X: {X.shape}")
print(f"y: {y.shape}")


# Gradient Descent ---------------------------------------------------------------------------------
def compute_cost(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    theta: NDArray[np.float64],
) -> float:
    """
    Compute the linear regression cost function.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Design matrix, including the intercept column.
    y : ndarray of shape (n_samples,)
        Target values.
    theta : ndarray of shape (n_features,)
        Model coefficients.

    Returns
    -------
    float
        Half the mean squared error.
    """
    n_samples = y.shape[0]
    residuals = X @ theta - y
    return float((residuals @ residuals) / (2 * n_samples))


def gradient_descent(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    theta: NDArray[np.float64],
    alpha: float,
    n_iterations: int,
) -> tuple[NDArray[np.float64], list[float]]:
    """
    Minimize the linear regression cost function using gradient descent.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Design matrix, including the intercept column.
    y : ndarray of shape (n_samples,)
        Target values.
    theta : ndarray of shape (n_features,)
        Initial parameter values.
    alpha : float
        Learning rate.
    n_iterations : int
        Number of gradient descent steps.

    Returns
    -------
    tuple[ndarray, list[float]]
        Final parameter vector and cost history.
    """
    n_samples = y.shape[0]
    theta = theta.copy()
    cost_history: list[float] = []

    for i in range(n_iterations):
        residuals = X @ theta - y
        gradient = (X.T @ residuals) / n_samples
        theta -= alpha * gradient

        cost = compute_cost(X, y, theta)
        cost_history.append(cost)

        if i % 100 == 0:
            print(f"Iteration {i:4d}: Cost = {cost:.6f}")

    return theta, cost_history


def predict(
    X: NDArray[np.float64],
    theta: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Generate predictions from a fitted linear regression model.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Design matrix, including the intercept column.
    theta : ndarray of shape (n_features,)
        Model coefficients.

    Returns
    -------
    ndarray of shape (n_samples,)
        Predicted values.
    """
    return X @ theta


def r_squared(
    y_true: NDArray[np.float64],
    y_pred: NDArray[np.float64],
) -> float:
    """
    Compute the coefficient of determination.

    Parameters
    ----------
    y_true : ndarray of shape (n_samples,)
        Observed target values.
    y_pred : ndarray of shape (n_samples,)
        Predicted target values.

    Returns
    -------
    float
        R-squared value.
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1 - ss_res / ss_tot)


# Fit Model ----------------------------------------------------------------------------------------
n_features = X.shape[1]
theta_initial = np.zeros(n_features, dtype=float)
alpha = 0.01
n_iterations = 1000

theta_final, cost_history = gradient_descent(
    X=X,
    y=y,
    theta=theta_initial,
    alpha=alpha,
    n_iterations=n_iterations,
)

# Results ------------------------------------------------------------------------------------------
y_pred = predict(X, theta_final)
r2 = r_squared(y, y_pred)
final_cost = compute_cost(X, y, theta_final)

coef_names = ["intercept", *X_df.columns]
coef_df = pd.DataFrame(
    {
        "feature": coef_names,
        "coefficient": theta_final,
    }
)

print("\nFinal coefficients:")
print(coef_df)

print("\nModel performance:")
print(f"Final cost: {final_cost:.6f}")
print(f"R-squared:  {r2:.4f}")

# Plots --------------------------------------------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(cost_history)
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Gradient Descent Convergence")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.scatter(y, y_pred, alpha=0.3)
plt.xlabel("Observed target")
plt.ylabel("Predicted target")
plt.title("Observed vs Predicted")
plt.tight_layout()
plt.show()
