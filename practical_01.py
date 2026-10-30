import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Sample dataset
data = {
    'Study_Hours': [1, 2, 3, 4, 5, 6, 7, 8],
    'Exam_Marks': [35, 40, 50, 55, 65, 70, 80, 90]
}

# Create DataFrame
df = pd.DataFrame(data)

print("Dataset:")
print(df)

# Calculate correlation
correlation = df['Study_Hours'].corr(df['Exam_Marks'])

print("\nCorrelation:", round(correlation, 2))

# Identify association
if correlation > 0:
    print("Positive Association")
elif correlation < 0:
    print("Negative Association")
else:
    print("No Linear Association")

# Plot scatter graph
sns.scatterplot(
    x='Study_Hours',
    y='Exam_Marks',
    data=df
)

plt.title("Study Hours vs Exam Marks")
plt.xlabel("Study Hours (Independent Variable)")
plt.ylabel("Exam Marks (Dependent Variable)")
plt.show()