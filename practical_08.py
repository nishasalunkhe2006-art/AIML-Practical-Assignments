import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score

# Generate sample dataset
np.random.seed(42)
X = np.linspace(-3, 3, 40).reshape(-1, 1)
y = np.sin(X).ravel() + np.random.normal(0, 0.15, 40)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Create models using polynomial features
degree = 12

models = {
    "Without Regularization": make_pipeline(
        PolynomialFeatures(degree),
        StandardScaler(),
        LinearRegression()
    ),

    "Ridge Regularization": make_pipeline(
        PolynomialFeatures(degree),
        StandardScaler(),
        Ridge(alpha=10)
    ),

    "Lasso Regularization": make_pipeline(
        PolynomialFeatures(degree),
        StandardScaler(),
        Lasso(alpha=0.01, max_iter=100000)
    )
}

# Train and evaluate models
print("Regularization Results")
for name, model in models.items():
    model.fit(X_train, y_train)

    train_score = r2_score(y_train, model.predict(X_train))
    test_score = r2_score(y_test, model.predict(X_test))

    print("\n", name)
    print("Training R2 Score:", round(train_score, 3))
    print("Testing R2 Score:", round(test_score, 3))

# Generate predictions for graph
X_plot = np.linspace(-3, 3, 200).reshape(-1, 1)

plt.figure(figsize=(9, 6))
plt.scatter(X, y, color="black", label="Actual Data")

for name, model in models.items():
    plt.plot(X_plot, model.predict(X_plot), label=name)

plt.title("Regularization to Avoid Overfitting")
plt.xlabel("Input Feature")
plt.ylabel("Target Value")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save graph in the same folder as this Python file
graph_path = Path(__file__).resolve().parent / "practical_08_graph.png"
plt.savefig(graph_path)

print("\nGraph saved at:", graph_path)
plt.show()