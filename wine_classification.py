import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Load wine data as a DataFrame
wines = load_wine(as_frame=True)["frame"]

# Select features and target
X = wines.drop("target", axis=1)
y = wines.target

# Split train/test data (40% test data)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42
)

# Create and fit model
model = GaussianNB()
model.fit(X_train, y_train)

# Predict and score accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
