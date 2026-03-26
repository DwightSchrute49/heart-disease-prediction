import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib


# Load Dataset

df = pd.read_csv("heart.csv")

print("Dataset Loaded Successfully\n")
print("First 5 rows:\n")
print(df.head())

print("\nColumns:\n", df.columns)


# Prepare Data


# Target column is "Heart Disease"
X = df.drop("Heart Disease", axis=1)
y = df["Heart Disease"]

# Convert target from text to numbers
y = y.map({"Presence": 1, "Absence": 0})


# Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Feature Scaling

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Train Model

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)


# Predictions

y_pred = model.predict(X_test)


# Evaluation

print("\nModel Evaluation\n")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report-\n")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# save model
joblib.dump(model, "heart_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nModel and scaler saved successfully!!..")

print("Training Accuracy:", model.score(X_train, y_train))
print("Testing Accuracy:", model.score(X_test, y_test))