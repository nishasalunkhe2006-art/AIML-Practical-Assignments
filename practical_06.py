import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Sample dataset
study_hours = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
exam_marks = [35, 40, 50, 55, 65, 70, 75, 85, 90, 95]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    study_hours, exam_marks, test_size=0.2, random_state=42
)

# Create and train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict marks
y_pred = model.predict(X_test)

# Predict marks for 6 study hours
new_prediction = model.predict([[6]])

# Display results
print("Linear Regression Results")
print("Predicted Marks for 6 Study Hours:", round(new_prediction[0], 2))
print("Mean Squared Error:", round(mean_squared_error(y_test, y_pred), 2))
print("R2 Score:", round(r2_score(y_test, y_pred), 2))

# Plot regression line
plt.scatter(study_hours, exam_marks, color="blue", label="Actual Data")
plt.plot(study_hours, model.predict(study_hours),
         color="red", label="Regression Line")

plt.title("Linear Regression - Study Hours vs Exam Marks")
plt.xlabel("Study Hours (Independent Variable)")
plt.ylabel("Exam Marks (Dependent Variable)")
plt.legend()
plt.grid(True)

# Save graph
plt.savefig("practical_06_graph.png")
plt.show()