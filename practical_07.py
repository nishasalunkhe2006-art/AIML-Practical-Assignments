import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Sample dataset
study_hours = [[1], [2], [3], [4], [5],
               [6], [7], [8], [9], [10]]

exam_marks = [35, 40, 50, 55, 65, 70, 75, 85, 90, 95]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    study_hours, exam_marks, test_size=0.2, random_state=42
)

# Scale input features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train SVR model
model = SVR(kernel="rbf", C=100, epsilon=0.1)
model.fit(X_train_scaled, y_train)

# Predict test data
y_pred = model.predict(X_test_scaled)

# Predict marks for 6 study hours
new_input = scaler.transform([[6]])
prediction = model.predict(new_input)

# Display results
print("Support Vector Regression Results")
print("Predicted Marks for 6 Study Hours:", round(prediction[0], 2))
print("Mean Squared Error:", round(mean_squared_error(y_test, y_pred), 2))
print("R2 Score:", round(r2_score(y_test, y_pred), 2))

# Plot SVR regression curve
X_plot = scaler.transform(study_hours)
y_plot = model.predict(X_plot)

plt.scatter(study_hours, exam_marks, color="blue", label="Actual Data")
plt.plot(study_hours, y_plot, color="red", label="SVR Prediction")

plt.title("Support Vector Regression")
plt.xlabel("Study Hours")
plt.ylabel("Exam Marks")
plt.legend()
plt.grid(True)

# Save graph
plt.savefig("practical_07_graph.png")
plt.show()