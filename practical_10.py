import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from scipy.sparse import hstack, csr_matrix

# Step 1: Create sample multimodal dataset
texts = [
    "Government announces new education policy",
    "Weather department issues rainfall warning",
    "Scientists publish new research findings",
    "School examination timetable released",
    "Government launches public health programme",
    "Official report confirms economic growth",
    "Hospital announces free health checkup",
    "University opens admission applications",
    "Breaking miracle cure works instantly",
    "Celebrity claims unbelievable secret formula",
    "Shocking news says aliens attacked the city",
    "Fake message promises instant lottery prize",
    "Viral post claims impossible scientific discovery",
    "Secret medicine cures every disease",
    "False message says banks give free money",
    "Unverified post claims celebrity is missing"
]

labels = [
    "Real", "Real", "Real", "Real",
    "Real", "Real", "Real", "Real",
    "Fake", "Fake", "Fake", "Fake",
    "Fake", "Fake", "Fake", "Fake"
]

# Step 2: Generate sample images
image_features = []

for i, label in enumerate(labels):
    if label == "Real":
        base_color = [40, 150, 80]
    else:
        base_color = [200, 40, 40]

    image_array = np.zeros((64, 64, 3), dtype=np.uint8)

    for channel in range(3):
        noise = np.random.randint(0, 30, (64, 64))
        image_array[:, :, channel] = np.clip(
            base_color[channel] + noise, 0, 255
        )

    img = Image.fromarray(image_array)

    # Extract image features
    resized_img = img.resize((32, 32))
    image_array = np.array(resized_img)

    mean_colors = image_array.mean(axis=(0, 1))
    image_features.append(mean_colors)

image_features = np.array(image_features)

# Step 3: Split data into training and testing
X_train_text, X_test_text, X_train_img, X_test_img, y_train, y_test = train_test_split(
    texts,
    image_features,
    labels,
    test_size=0.25,
    random_state=42,
    stratify=labels
)

# Step 4: Convert text into numerical features
vectorizer = TfidfVectorizer()
X_train_text = vectorizer.fit_transform(X_train_text)
X_test_text = vectorizer.transform(X_test_text)

# Step 5: Combine text and image features
X_train = hstack([
    X_train_text,
    csr_matrix(X_train_img)
])

X_test = hstack([
    X_test_text,
    csr_matrix(X_test_img)
])

# Step 6: Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Step 7: Make predictions
y_pred = model.predict(X_test)

print("MULTIMODAL FAKE NEWS DETECTION")
print("--------------------------------")
print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Step 8: Display confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=["Fake", "Real"])

plt.imshow(cm, cmap="Blues")
plt.title("Fake News Detection")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.xticks([0, 1], ["Fake", "Real"])
plt.yticks([0, 1], ["Fake", "Real"])

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.colorbar()
plt.tight_layout()
plt.savefig("practical_10_graph.png")
plt.show()

print("\nModel execution completed successfully!")