# Setup --------------------------------------------------------------------------------------------
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, export_text, plot_tree

# Load data ----------------------------------------------------------------------------------------
# Download data and save it if csv file does not exist, otherwise load from csv file
# The data contains information about baseball players: salaries and various performance metrics.
data_path = Path("data/hitters.csv")

if data_path.exists():
    df = pd.read_csv(data_path)
else:
    url = (
        "https://raw.githubusercontent.com/JWarmenhoven/ISLR-python/master/Notebooks/Data/"
        "Hitters.csv"
    )
    df = pd.read_csv(url)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(data_path, index=False)

# Keep only variables needed here and remove missing values ----------------------------------------
df = df[["Salary", "Hits", "Years"]].dropna().copy()

X = df[["Hits", "Years"]]
y = df["Salary"]

# Visualization ------------------------------------------------------------------------------------
# Years vs Hits scatter plot with color representing Salary
fig, ax = plt.subplots(figsize=(10, 6))
sc = ax.scatter(X["Years"], X["Hits"], c=y, cmap="viridis", edgecolors="w", s=90)
ax.set(xlabel="Years", ylabel="Hits", title="Years vs Hits (color = Salary)")
fig.colorbar(sc, ax=ax, label="Salary")
plt.show()

# Fit a simple regression tree ---------------------------------------------------------------------
tree = DecisionTreeRegressor(
    max_depth=20,
    min_samples_leaf=30,
    random_state=42,
)
tree.fit(X, y)

# Inspect predictions ------------------------------------------------------------------------------
df["pred_salary"] = tree.predict(X)

print(df.head())
print(f"Number of leaves: {tree.get_n_leaves()}")
print(f"Tree depth: {tree.get_depth()}")

# Print the tree as text ---------------------------------------------------------------------------
tree_rules = export_text(tree, feature_names=list(X.columns))
print(tree_rules)

# Plot the tree ------------------------------------------------------------------------------------
plt.figure(figsize=(10, 6))
plot_tree(
    tree,
    feature_names=X.columns,
    filled=True,
    rounded=True,
    impurity=True,
)
plt.tight_layout()
plt.show()
